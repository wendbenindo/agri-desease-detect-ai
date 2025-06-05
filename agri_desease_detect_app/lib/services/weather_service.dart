import 'dart:convert';
import 'package:agri_desease_detect_app/model/weather_model.dart';
import 'package:http/http.dart' as http;
import 'package:geolocator/geolocator.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import '../utils/constants.dart';

class WeatherService {
  Future<WeatherModel> fetchWeather() async {
    final Position position = await _determinePosition();
    final apiKey = dotenv.env['WEATHER_API_KEY'];
    final url = Uri.parse('$baseWeatherUrl?lat=...&appid=$apiKey');


    final response = await http.get(url);

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

    return await Geolocator.getCurrentPosition();
  }
}
