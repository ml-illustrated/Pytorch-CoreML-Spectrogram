//
//  ViewController.swift
//  Pytorch-CoreML-Spectrogram
//
//  Created by Gerald on 5/24/20.
//  Copyright © 2020 Gerald. All rights reserved.
//

// reference: https://developer.apple.com/documentation/speech/recognizing_speech_in_live_audio

import UIKit
import AVKit
import CoreML

class ViewController: UIViewController {

    @IBOutlet weak var drawSpecView: DrawSpecView!
    
    // set up for audio
    private let audioEngine = AVAudioEngine()
    // specify the audio samples format the CoreML model
    let desiredAudioFormat: AVAudioFormat = {
        let avAudioChannelLayout = AVAudioChannelLayout(layoutTag: kAudioChannelLayoutTag_Mono)!
        return AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double( 32000 ), // as specified when creating the Pytorch model
            interleaved: true,
            channelLayout: avAudioChannelLayout
        )
    }()
    
    // create a queue to do analysis on a separate thread
    let analysisQueue = DispatchQueue(label: "com.myco.AnalysisQueue")
    
    // instantiate our model
    let model = wave__melspec()
    typealias NetworkInput = wave__melspecInput
    typealias NetworkOutput = wave__melspecOutput

    // semaphore to protect the CoreML model
    let semaphore = DispatchSemaphore(value: 1)

    // for rendering our spectrogram
    let spec_converter = SpectrogramConverter()
  
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }
    
    override func viewDidAppear(_ animated: Bool) {
        startAudioEngine()
    }
    
    // audio capture via microphone
    private func startAudioEngine() {
        
        // https://stackoverflow.com/questions/48831411/converting-avaudiopcmbuffer-to-another-avaudiopcmbuffer
        // more info at https://medium.com/@prianka.kariat/changing-the-format-of-ios-avaudioengine-mic-input-c183459cab63
        
        let inputNode = audioEngine.inputNode
        let originalAudioFormat: AVAudioFormat = inputNode.inputFormat(forBus: 0)
        // input is in 44.1kHz, 2 channels

        let downSampleRate: Double = desiredAudioFormat.sampleRate
        let ratio: Float = Float(originalAudioFormat.sampleRate)/Float(downSampleRate)

        // print( "input sr: \(originalAudioFormat.sampleRate) ch: \(originalAudioFormat.channelCount)" )
        // print( "desired sr: \(desiredAudioFormat.sampleRate) ch: \(desiredAudioFormat.channelCount) ratio \(ratio)" )
        
        guard let formatConverter =  AVAudioConverter(from:originalAudioFormat, to: desiredAudioFormat) else {
            fatalError( "unable to create formatConverter!" )
        }

        // start audio capture by installing a Tap
        inputNode.installTap(
            onBus: 0,
            bufferSize: AVAudioFrameCount(downSampleRate * 2),
            format: originalAudioFormat
        ) {
            (buffer: AVAudioPCMBuffer!, time: AVAudioTime!) in
            // closure to process the captured audio, buffer size dictated by AudioEngine/device
             
            let capacity = UInt32(Float(buffer.frameCapacity)/ratio)

            guard let pcmBuffer = AVAudioPCMBuffer(
                pcmFormat: self.desiredAudioFormat,
                frameCapacity: capacity) else {
              print("Failed to create pcm buffer")
              return
            }
            
            let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
              outStatus.pointee = AVAudioConverterInputStatus.haveData
              return buffer
            }

            // convert input samples into the one our model needs
            var error: NSError?
            let status: AVAudioConverterOutputStatus = formatConverter.convert(
                to: pcmBuffer,
                error: &error,
                withInputFrom: inputBlock)

            if status == .error {
                if let unwrappedError: NSError = error {
                    print("Error \(unwrappedError)")
                }
                return
            }
            
            // we now have the audio in mono, 32000 sample rate the CoreML model needs
            // convert audio samples into MLMultiArray format for CoreML models
            let channelData = pcmBuffer.floatChannelData
            let output_samples = Int(pcmBuffer.frameLength)
            let channelDataPointer = channelData!.pointee
            
            // print( "converted from \(buffer.frameLength) to len \(output_samples) val[0] \(channelDataPointer[0]) \(channelDataPointer[output_samples-1])" )

            let audioData = try! MLMultiArray( shape: [1, output_samples as NSNumber], dataType: .float32 )
            let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(audioData.dataPointer))
            for i in 0..<output_samples {
                ptr[i] = Float32(channelDataPointer[i])
            }

            // prepare the input dictionary
            let inputs: [String: Any] = [
                "input.1": audioData,
            ]
            // container for ML Model inputs
            let provider = try! MLDictionaryFeatureProvider(dictionary: inputs)
                   
            // wait in case CoreML model is busy
            self.semaphore.wait()

            self.analysisQueue.async {
                // send this sample to CoreML to generate melspectrogram
                self.predict_provider(provider: provider)
            }
        } // installTap

        // ready to start the actual audio capture
        audioEngine.prepare()
        do {
            try audioEngine.start()
        }
        catch {
           print(error.localizedDescription)
        }
    } // end startAudioEngine
    
    
    func predict_provider(provider: MLDictionaryFeatureProvider ) {
        if let outFeatures = try? self.model.model.prediction(from: provider) {
            // release the semaphore as soon as the model is done
            self.semaphore.signal()

            let outputs = NetworkOutput(features: outFeatures)
            let output_spectrogram: MLMultiArray = outputs._25

            // melspectrogram is in MLMultiArray in decibels. Convert to 0..1 for visualization
            // and then pass the converted spectrogram to the UI element drawSpecView
            drawSpecView.spectrogram = spec_converter.convertTo2DArray(from: output_spectrogram)
        } else {
            self.semaphore.signal()
        }
    }


}
