import 'dart:io';
import 'package:flutter/material.dart';

class ImageAnalysisModule extends StatelessWidget {
  final File image;

  const ImageAnalysisModule({super.key, required this.image});

  @override
  Widget build(BuildContext context) {
    // Ce module pourrait appeler une IA, un modèle ML ou une API
    return Scaffold(
      appBar: AppBar(
        title: const Text('Analyse en cours'),
        backgroundColor: Colors.black,
      ),
      backgroundColor: Colors.white,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Image.file(image, width: 200),
            const SizedBox(height: 24),
            const CircularProgressIndicator(),
            const SizedBox(height: 16),
            const Text(
              'Analyse en cours...\nVeuillez patienter.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }
}
