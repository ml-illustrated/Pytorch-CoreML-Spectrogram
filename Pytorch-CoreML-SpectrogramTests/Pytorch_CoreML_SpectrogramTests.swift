//
//  Pytorch_CoreML_SpectrogramTests.swift
//  Pytorch-CoreML-SpectrogramTests
//
//  Created by Gerald on 5/31/20.
//  Copyright © 2020 Gerald. All rights reserved.
//

import XCTest
import AVFoundation
import CoreML

@testable import Pytorch_CoreML_Spectrogram

class Pytorch_CoreML_SpectrogramTests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func test_wav__spectrogram() {
        // instantiate our CoreML model
        let model = wave__melspec()
        typealias NetworkInput = wave__melspecInput
        typealias NetworkOutput = wave__melspecOutput

        // read in the expected model output from JSON
        let bundle = Bundle(for: Pytorch_CoreML_SpectrogramTests.self)
        let path = bundle.path(forResource: "melspec_out.bonjour", ofType: "json")
        let data = try! Data(contentsOf: URL(fileURLWithPath: path!))
        let expected_spectrogram: [[NSNumber]] = try! JSONSerialization.jsonObject(with: data) as! [[NSNumber]]

        print( "expected spec: \(expected_spectrogram.count) \(expected_spectrogram[0].count)")

        // read the input shapes of our model
        let inputName = "input.1"
        let inputConstraint: MLFeatureDescription = model.model.modelDescription
            .inputDescriptionsByName[inputName]!

        let input_batch_size: Int = Int(truncating: (inputConstraint.multiArrayConstraint?.shape[0])! )
        let input_samples: Int = Int(truncating: (inputConstraint.multiArrayConstraint?.shape[1])! )
        // print( "inputCon: \(input_batch_size) \(input_samples)")

        // read the same WAV file used in PyTorch
        let testBundle = Bundle(for: type(of: self))
        guard let filePath = testBundle.path(forResource: "bonjour", ofType: "wav") else {
            fatalError( "error opening bonjour.wav" )
        }
        
        // Read wav file
        var wav_file:AVAudioFile!
        do {
           let fileUrl = URL(fileURLWithPath: filePath )
           wav_file = try AVAudioFile( forReading:fileUrl )
        } catch {
           fatalError("Could not open wav file.")
        }

        let buffer = AVAudioPCMBuffer(pcmFormat: wav_file.processingFormat,
                                      frameCapacity: UInt32(wav_file.length))
        do {
           try wav_file.read(into:buffer!)
        } catch{
           fatalError("Error reading buffer.")
        }
        
        guard let bufferData = buffer?.floatChannelData![0] else {
           fatalError("Can not get a float handle to buffer")
        }
            

        // allocate a ML Array & copy samples over
        let array_shape = [input_batch_size as NSNumber, input_samples as NSNumber]
        let audioData = try! MLMultiArray(shape: array_shape, dataType: MLMultiArrayDataType.float32 )
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(audioData.dataPointer))
        for i in 0..<input_samples {
            ptr[i] = Float32(bufferData[i])
        }

        // create the input dictionary as { 'input.1' : [<wave floats>] }
        let inputs: [String: Any] = [
            inputName: audioData,
        ]
        // container for ML Model inputs
        let provider = try! MLDictionaryFeatureProvider(dictionary: inputs)
        
        // Send the wavefor samples into the model to generate the Spectrogram
        let raw_outputs = try! model.model.prediction(from: provider)
        
        // convert raw dictionary into our model's output object
        let outputs = NetworkOutput( features: raw_outputs )
        // the output we're interested in is "_25"
        let output_spectrogram: MLMultiArray = outputs._25
        print( "outputs: \(output_spectrogram.shape)") // [1, 1, 41, 64]

        // sanity check the shapes of our output
        XCTAssertTrue( Int( truncating: output_spectrogram.shape[2] ) == expected_spectrogram.count,
                       "incorrect shape[2]! \(output_spectrogram.shape[2]) \(expected_spectrogram.count)" )
        XCTAssertTrue( Int( truncating: output_spectrogram.shape[3] ) == expected_spectrogram[0].count,
                       "incorrect shape[3]! \(output_spectrogram.shape[3]) \(expected_spectrogram[0].count)" )

        // compare every element of our spectrogram with those from the JSON file
        for i in 0..<expected_spectrogram.count {
            let spec_row = expected_spectrogram[i] as [NSNumber]

            for j in 0..<spec_row.count {
                let test_idx: [NSNumber] = [ 0, 0, NSNumber(value: i), NSNumber(value: j) ]
                let val = output_spectrogram[ test_idx ].floatValue
                XCTAssertLessThan( abs( val - spec_row[j].floatValue ), 0.01,
                                   "spec vals different at \(i) \(j)!" )
            }
        }
    }
    
    func test_convert_spectrogram() throws {
        let spec_converter = SpectrogramConverter()

        // load our spectrogram from JSON
        let bundle = Bundle(for: Pytorch_CoreML_SpectrogramTests.self)
        let path = bundle.path(forResource: "melspec_out.bonjour", ofType: "json")
        let data = try! Data(contentsOf: URL(fileURLWithPath: path!))
        let expected_spectrogram: [[NSNumber]] = try! JSONSerialization.jsonObject(with: data) as! [[NSNumber]]

        // copy data over to an allocated MLMultiArray
        let spec_cols = expected_spectrogram.count
        let spec_rows = expected_spectrogram[0].count
        let array_shape = [ 1, 1, spec_cols as NSNumber, spec_rows as NSNumber ]
        let spectrogram = try! MLMultiArray(shape: array_shape, dataType: MLMultiArrayDataType.float32 )
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(spectrogram.dataPointer))
        for i in 0..<expected_spectrogram.count {
            let spec_row = expected_spectrogram[i] as [NSNumber]

            for j in 0..<spec_row.count {
                ptr[ i*Int(spectrogram.strides[2]) + j] = Float32( spec_row[j] )
            }
        }
        // call convert
        let converted_spec = spec_converter.convertTo2DArray(from: spectrogram )
        
        // check shape, should be the same
        XCTAssertTrue( converted_spec.count == spec_cols, "converted spec shape incorrect!" )
        XCTAssertTrue( converted_spec[0].count == spec_rows, "converted spec shape incorrect!" )

        // spot check the converted specs
        XCTAssertTrue( converted_spec[0].min()! >= Float32(0.0), "converted spec min incorrect!" )
        XCTAssertTrue( converted_spec[0].max()! <= Float32(1.0), "converted spec max incorrect!" )

    }

    func test_inference_time() throws {
        // This is an example of a performance test case.
        let model = wave__melspec()

        let array_shape: [NSNumber] = [1, 12800]
        let audioData = try! MLMultiArray(shape: array_shape, dataType: MLMultiArrayDataType.float32 )
        let inputs: [String: Any] = [
            "input.1": audioData,
        ]
        // container for ML Model inputs
        let provider = try! MLDictionaryFeatureProvider(dictionary: inputs)

        self.measure {
            // Put the code you want to measure the time of here.
            let N = 100
            let start_time = CACurrentMediaTime()
            let options = MLPredictionOptions()
            // options.usesCPUOnly = true
            for _ in 0..<N {
                _ = try? model.model.prediction(
                    from: provider,
                    options: options
                )
            }
            let elapsed = CACurrentMediaTime() - start_time
            print( "avg inference time: \(elapsed/Double(N))")
            /* simulator:
                avg inference time: 0.011592097150278279
                w/ CPUOnly: avg inference time: 0.011968133399495855
               on iPhone XR
                avg inference time: 0.003219694583094679
                w/ CPUOnly: avg inference time: 0.003034873333526775

             */
        }
    }

}
