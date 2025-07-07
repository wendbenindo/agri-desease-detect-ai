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
  String _result = '';
  bool _isLoading = true;
  int _currentStep = 0;

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
    _startAnalysisProcess();
  }

  Future<void> _startAnalysisProcess() async {
    await Future.delayed(const Duration(milliseconds: 500));
    setState(() => _currentStep = 1);
    await Future.delayed(const Duration(milliseconds: 500));
    await _analyzeImage();
    setState(() => _currentStep = 3);
  }

  Future<void> _analyzeImage() async {
    try {
      final interpreter = await Interpreter.fromAsset('assets/model/plant_disease_model.tflite');

      final imageBytes = await widget.image.readAsBytes();
      img.Image? oriImage = img.decodeImage(imageBytes);
      if (oriImage == null) throw Exception("Image illisible");

      setState(() => _currentStep = 2);

      img.Image resizedImage = img.copyResize(oriImage, width: 224, height: 224);

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

      final inputBuffer = input.buffer.asFloat32List().reshape([1, 224, 224, 3]);
      final output = List.filled(_classNames.length, 0.0).reshape([1, _classNames.length]);

      interpreter.run(inputBuffer, output);
      interpreter.close();

      final result = output[0] as List<double>;
      final maxProb = result.reduce((a, b) => a > b ? a : b);
      final predictedIndex = result.indexOf(maxProb);
      final predictedClass = _classNames[predictedIndex];

      setState(() {
        _result = '🌿 Classe prédite : $predictedClass\n🔬 Confiance : ${(maxProb * 100).toStringAsFixed(2)}%';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _result = 'Erreur pendant l’analyse : $e';
        _isLoading = false;
      });
    }
  }

  Widget _buildStep(int step, String label) {
    bool isCompleted = _currentStep > step;
    bool isCurrent = _currentStep == step;

    return Row(
      children: [
        Icon(
          isCompleted
              ? Icons.check_circle
              : isCurrent
                  ? Icons.autorenew
                  : Icons.radio_button_unchecked,
          color: const Color(0xFF15803D),
        ),
        const SizedBox(width: 8),
        Text(label, style: const TextStyle(fontSize: 14, color: Colors.black87)),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Analyse de l’image',
            style: TextStyle(color: Color(0xFF15803D), fontWeight: FontWeight.bold)),
        backgroundColor: Colors.white,
        iconTheme: const IconThemeData(color: Color(0xFF15803D)),
        centerTitle: true,
        elevation: 1,
      ),
      backgroundColor: Colors.white,
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Image.file(widget.image, height: 250, fit: BoxFit.cover),
            ),
            const SizedBox(height: 24),
            const Text('Étapes de traitement',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 12),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildStep(0, 'Chargement de l’image'),
                _buildStep(1, 'Préparation des données'),
                _buildStep(2, 'Exécution du modèle'),
                _buildStep(3, 'Affichage du résultat'),
              ],
            ),
            const SizedBox(height: 24),
            if (_isLoading)
              const CircularProgressIndicator(color: Color(0xFF15803D))
            else
              Container(
                padding: const EdgeInsets.all(16),
                width: double.infinity,
                decoration: BoxDecoration(
                  color: const Color(0xFFF0FDF4),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: const Color(0xFF15803D).withOpacity(0.4)),
                ),
                child: Text(
                  _result,
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: Colors.black87),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
