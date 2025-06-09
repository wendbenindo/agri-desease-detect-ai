import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class ImageAnalysisModule extends StatefulWidget {
  final File image;

  const ImageAnalysisModule({super.key, required this.image});

  @override
  State<ImageAnalysisModule> createState() => _ImageAnalysisModuleState();
}

class _ImageAnalysisModuleState extends State<ImageAnalysisModule> {
  String _result = 'Analyse en cours...';
  bool _isLoading = true;

  final List<String> _classNames = [
    'Healthy',
    'Maize Leaf Spot',
    'Maize Streak',
    'Mil Sorgho',
    'Sorghum Blight',
    'Sorghum Rust',
  ];

  @override
  void initState() {
    super.initState();
    _analyzeImage();
  }

  Future<void> _analyzeImage() async {
    try {
      final interpreter = await Interpreter.fromAsset('assets/model/plant_disease_model.tflite');

      final imageBytes = await widget.image.readAsBytes();
      img.Image? oriImage = img.decodeImage(imageBytes);
      if (oriImage == null) {
        setState(() {
          _result = 'Erreur : image illisible.';
          _isLoading = false;
        });
        return;
      }

      // Redimensionnement à 224x224 (MobileNet input size)
      img.Image resizedImage = img.copyResize(oriImage, width: 224, height: 224);

      // Transformation image en Float32List normalisée [0, 1]
      Float32List input = Float32List(224 * 224 * 3);
      int index = 0;

      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          final pixel = resizedImage.getPixel(x, y);
          input[index++] = img.getRed(pixel) / 255.0;
          input[index++] = img.getGreen(pixel) / 255.0;
          input[index++] = img.getBlue(pixel) / 255.0;
        }
      }

      // Formater en [1, 224, 224, 3] pour l'entrée
      final inputShape = interpreter.getInputTensor(0).shape;
      final inputTensor = input.buffer.asFloat32List();
      final inputBuffer = inputTensor.reshape([1, 224, 224, 3]);

      final output = List.filled(_classNames.length, 0.0).reshape([1, _classNames.length]);

      interpreter.run(inputBuffer, output);

      final result = output[0] as List<double>;
      final maxProb = result.reduce((a, b) => a > b ? a : b);
      final predictedIndex = result.indexOf(maxProb);
      final predictedClass = _classNames[predictedIndex];

      setState(() {
        _result = 'Résultat : $predictedClass\nConfiance : ${(maxProb * 100).toStringAsFixed(2)}%';
        _isLoading = false;
      });

      interpreter.close();
    } catch (e) {
      setState(() {
        _result = 'Erreur pendant l’analyse : $e';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Analyse de l’image'),
        backgroundColor: Colors.black,
      ),
      backgroundColor: Colors.white,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Image.file(widget.image, width: 250),
            const SizedBox(height: 24),
            _isLoading
                ? const CircularProgressIndicator()
                : Text(
                    _result,
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
                  ),
          ],
        ),
      ),
    );
  }
}
