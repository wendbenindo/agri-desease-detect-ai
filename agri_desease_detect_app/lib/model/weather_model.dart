class WeatherModel {
  final double temperature;
  final double feelsLike;
  final String condition;
  final String description;
  final String city;
  final String country;
  final double windSpeed;
  final int humidity;
  final int visibility;

  WeatherModel({
    required this.temperature,
    required this.feelsLike,
    required this.condition,
    required this.description,
    required this.city,
    required this.country,
    required this.windSpeed,
    required this.humidity,
    required this.visibility,
  });

  factory WeatherModel.fromJson(Map<String, dynamic> json) {
    return WeatherModel(
      temperature: (json['main']?['temp'] ?? 0).toDouble(),
      feelsLike: (json['main']?['feels_like'] ?? 0).toDouble(),
      condition: json['weather']?[0]?['main'] ?? 'Inconnu',
      description: json['weather']?[0]?['description'] ?? 'N/A',
      city: json['name'] ?? 'Ville inconnue',
      country: json['sys']?['country'] ?? '--',
      windSpeed: (json['wind']?['speed'] ?? 0).toDouble(),
      humidity: json['main']?['humidity'] ?? 0,
      visibility: json['visibility'] ?? 0,
    );
  }
}
