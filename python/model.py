import numpy as np
import torch
from torch import nn
import torchlibrosa

class WaveToLogmelSpectrogram(nn.Module):

    def __init__(
            self,
            sample_rate=32000,
            n_fft=1024,
            hop_length=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
    ):
        super(WaveToLogmelSpectrogram, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        
        self.spec_extractor = torchlibrosa.stft.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length, 
            win_length=n_fft,
            window=window,
            center=center,
            pad_mode=pad_mode, 
            freeze_parameters=True
        )

        self.logmel_extractor = torchlibrosa.stft.LogmelFilterBank(
            sr=sample_rate,
            n_fft=n_fft, 
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db, 
            freeze_parameters=True
        )
        

        self.input_name = 'input.1'
        self.output_name = '25' # looked up via Netron
    
    def forward( self, x ):
        x = self.spec_extractor( x )
        return self.logmel_extractor( x )

    def gen_torch_output( self, sample_input ):
        self.eval()
        with torch.no_grad():
            torch_output = self( torch.from_numpy( sample_input ) )
            torch_output = torch_output.cpu().detach().numpy()
        return torch_output

    def convert_to_onnx( self, filename_onnx, sample_input ):

        input_names = [ self.input_name ]
        output_names = [ self.output_name ]
        
        torch.onnx.export(
            self,
            torch.from_numpy( sample_input ),
            filename_onnx,
            input_names=input_names,
            output_names=output_names,
            # operator_export_type=OperatorExportTypes.ONNX
        )

    def gen_onnx_output( self, filename_onnx, sample_input ):
        import onnxruntime
        
        session = onnxruntime.InferenceSession( filename_onnx, None)
        
        input_name = session.get_inputs()[0].name
        # output_names = [ item.name for item in session.get_outputs() ]

        raw_result = session.run([], {input_name: sample_input})

        return raw_result[0]
        
        
    def convert_to_coreml( self, fn_mlmodel, sample_input, plot_specs=True ):
        import onnx
        import onnx_coreml
        
        torch_output = self.gen_torch_output( sample_input )
        # print( 'torch_output: shape %s\nsample %s ' % ( torch_output.shape, torch_output[:, :, :3, :3] ) )
        print( 'torch_output: shape ', ( torch_output.shape ) ) # (1, 1, 28, 64)

        # first convert to ONNX
        filename_onnx = '/tmp/wave__melspec_model.onnx'
        model.convert_to_onnx( filename_onnx, sample_input )

        onnx_output = self.gen_onnx_output( filename_onnx, sample_input )

        # set up for Core ML export
        convert_params = dict(
            predicted_feature_name = [],
            minimum_ios_deployment_target='13',
        )

        mlmodel = onnx_coreml.convert(
            model=filename_onnx,
            **convert_params, 
        )

        assert mlmodel != None, 'CoreML Conversion failed'

        mlmodel.save( fn_mlmodel )

        model_inputs = {
            self.input_name : sample_input
        }
        # do forward pass
        mlmodel_outputs = mlmodel.predict(model_inputs, useCPUOnly=True)

        # fetch the spectrogram from output dictionary
        mlmodel_output =  mlmodel_outputs[ self.output_name ]
        # print( 'mlmodel_output: shape %s \nsample %s ' % ( mlmodel_output.shape, mlmodel_output[:,:,:3, :3] ) )
        print( 'mlmodel_output: shape ', ( mlmodel_output.shape ) )

        assert torch_output.shape == mlmodel_output.shape

        print( 'sum diff ', np.sum( np.abs( torch_output-mlmodel_output) ), np.max( np.abs( torch_output-mlmodel_output) ) )
        assert np.allclose( torch_output, mlmodel_output, atol=2, rtol=2 ) # big tolerance due to log scale
            
        print( 'Successful MLModel conversion to %s!' % fn_mlmodel )

        if plot_specs:
            plot_spectrograms( torch_output, onnx_output, mlmodel_output )
    
        return mlmodel_output

def load_wav_file( fn_wav ):
    import soundfile as sf

    data, samplerate = sf.read( fn_wav )
    return data
    
def save_ml_model_output_as_json( fn_output, mlmodel_output ):
    import json
    with open( fn_output, 'w' ) as fp:
        json.dump( mlmodel_output.tolist(), fp )

def plot_spectrograms( torch_output, onnx_output, mlmodel_output ):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    def spec__image( spectrogram ):
        return spectrogram[0,0,...].T
        
    fig = plt.figure( figsize=(8,8) )
    
    a = fig.add_subplot(3, 1, 1)
    a.imshow( spec__image( torch_output ), aspect='auto', origin='lower', cmap='jet')
    a.set_title( 'Pytorch' )
    a.tick_params( axis='x', which='both', bottom=False, top=False, labelbottom=False)

    a = fig.add_subplot(3, 1, 2)
    a.imshow( spec__image( onnx_output ), aspect='auto', origin='lower', cmap='jet')
    a.set_title( 'ONNX' )
    a.tick_params( axis='x', which='both', bottom=False, top=False, labelbottom=False)

    a = fig.add_subplot(3, 1, 3)
    a.imshow( spec__image( mlmodel_output ), aspect='auto', origin='lower', cmap='jet')
    a.set_title( 'Core ML' )
    
    plt.show()
    
    
if __name__ == '__main__':
    import sys
    fn_sample_wav = sys.argv[1]
    fn_mlmodel = sys.argv[2]
    fn_model_output = sys.argv[3]

    num_samples = 12800 # hack, load samples same length as iOS audio buffer
    waveform = load_wav_file( fn_sample_wav )
    sample_input = waveform[ :num_samples ].astype( dtype=np.float32 )
    # shape: (samples_num,)
    
    # add batch dimension
    sample_input = np.expand_dims( sample_input, axis=0 )
    # shape: (batch_size, samples_num)
    
    model =  WaveToLogmelSpectrogram()

    # filename_onnx = '/tmp/wave__sound_events_model.onnx'
    # model.convert_to_onnx( filename_onnx, sample_input )
    
    mlmodel_output = model.convert_to_coreml( fn_mlmodel, sample_input )
    # shape: ??

    save_ml_model_output_as_json( fn_model_output, mlmodel_output[0,0,...])

'''
# example command:
python model.py ../Pytorch-CoreML-SpectrogramTests/bonjour.wav /tmp/wave__melspec.mlmodel  /tmp/melspec_out.bonjour.json
'''    

