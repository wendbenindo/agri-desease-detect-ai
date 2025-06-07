import 'dart:convert';
import 'package:agri_desease_detect_app/model/weather_model.dart';
import 'package:http/http.dart' as http;
import 'package:geolocator/geolocator.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import '../utils/constants.dart';

class WeatherService {
  Future<WeatherModel> fetchWeather() async {
    final Position position = await _determinePosition();

    //  Lire la clé API depuis le fichier .env
    final apiKey = dotenv.env['WEATHER_API_KEY'];
    if (apiKey == null || apiKey.isEmpty) {
      throw Exception('Clé API météo non définie.');
    }

    //  Construire une URL correcte avec lat/lon
    final url = Uri.parse(
      '$baseWeatherUrl?lat=${position.latitude}&lon=${position.longitude}&appid=$apiKey&units=metric&lang=fr',
    );

    //  Debug : afficher l’URL
    print("📡 Appel météo : $url");

    final response = await http.get(url);

    //  Debug : afficher code retour + contenu
    print("Code HTTP : ${response.statusCode}");
    print("Réponse : ${response.body}");

    if (response.statusCode == 200) {
      final json = jsonDecode(response.body);
      return WeatherModel.fromJson(json);
    } else {
      throw Exception('Erreur lors de la récupération de la météo');
    }
  }

  Future<Position> _determinePosition() async {
    bool serviceEnabled;
    LocationPermission permission;

    //  Vérifie si le service de localisation est activé
    serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      throw Exception('Le service de localisation est désactivé.');
    }

    permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        throw Exception('La permission de localisation est refusée.');
      }
    }

    if (permission == LocationPermission.deniedForever) {
      throw Exception('La permission de localisation est définitivement refusée.');
    }

    // Récupération de la position
    final position = await Geolocator.getCurrentPosition();
    print(" Position récupérée : ${position.latitude}, ${position.longitude}");
    return position;
  }
}
