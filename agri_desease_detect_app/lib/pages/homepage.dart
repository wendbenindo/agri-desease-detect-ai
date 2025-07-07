import 'dart:io';
import 'package:agri_desease_detect_app/pages/diagnosticpage.dart';
import 'package:flutter/material.dart';
import 'package:carousel_slider/carousel_slider.dart';
import 'package:image_picker/image_picker.dart';
import 'package:agri_desease_detect_app/services/weather_service.dart';
import 'package:agri_desease_detect_app/model/weather_model.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with WidgetsBindingObserver {
  final List<Map<String, String>> cultures = [
    {
      'image': 'assets/images/mil.png',
      'nom': 'Mil',
      'description': 'Le mil est une céréale résistante à la sécheresse, cultivée dans les régions semi-arides.',
      'physionomie': 'Plante herbacée à tige dressée, feuilles linéaires longues et fines.',
      'conditions': 'Température idéale : 25-35°C. Sols sablonneux bien drainés.',
      'maladies': 'Charbon du mil, mildiou, insectes foreurs.',
      'tests': 'Observation des feuilles, reconnaissance par IA via photo.',
      'conseils': 'Rotation culturale, traitement antifongique, évitez les excès d’eau.'
    },
    {
      'image': 'assets/images/sorgho.png',
      'nom': 'Sorgho',
      'description': 'Plante céréalière utilisée pour l’alimentation, le fourrage et les biocarburants.',
      'physionomie': 'Feuilles larges, inflorescence dense, tige robuste.',
      'conditions': 'Températures élevées (30-38°C), résistant à la sécheresse.',
      'maladies': 'Rouille, anthracnose, pourriture des tiges.',
      'tests': 'Inspection visuelle, symptômes sur les feuilles et les tiges.',
      'conseils': 'Bonne rotation, semis espacés, variétés résistantes.'
    },
    {
      'image': 'assets/images/mais.png',
      'nom': 'Maïs',
      'description': 'Céréale majeure utilisée pour l’alimentation humaine, animale et l’industrie.',
      'physionomie': 'Grande tige, feuilles larges, épis bien visibles.',
      'conditions': 'Température idéale : 20-30°C. Sols riches, arrosage régulier.',
      'maladies': 'Mildiou, fusariose, tache foliaire.',
      'tests': 'Analyse des feuilles, coloration anormale, photos pour détection.',
      'conseils': 'Fertilisation équilibrée, surveillance des parasites.'
    },
  ];

  final List<Map<String, String>> maladies = [
    {'image': 'assets/images/mildiou.png', 'nom': 'Mildiou du maïs'},
    {'image': 'assets/images/rouille.png', 'nom': 'Rouille du sorgho'},
    {'image': 'assets/images/tache_feuille.png', 'nom': 'Tache foliaire'},
  ];

  WeatherModel? _weather;
  bool _loading = false;
  bool _weatherError = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _loadWeather();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      _loadWeather();
    }
  }

  Future<void> _takePictureAndNavigate() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.camera);

    if (pickedFile != null) {
      setState(() => _loading = true);
      await Future.delayed(const Duration(seconds: 2));
      setState(() => _loading = false);

      if (!mounted) return;
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => DiagnosticPage(initialImage: File(pickedFile.path)),
        ),
      );
    }
  }

  Future<void> _loadWeather() async {
    try {
      final weather = await WeatherService().fetchWeather();
      setState(() {
        _weather = weather;
        _weatherError = false;
      });
    } catch (e) {
      debugPrint('Erreur météo : $e');
      setState(() => _weatherError = true);
    }
  }

  Widget _weatherInfoTile(IconData icon, String value, String label) {
    return Column(
      children: [
        Icon(icon, size: 24, color: const Color(0xFF15803D)),
        const SizedBox(height: 4),
        Text(value, style: const TextStyle(fontWeight: FontWeight.bold)),
        Text(label, style: const TextStyle(fontSize: 12, color: Colors.grey)),
      ],
    );
  }

   void _showCultureDetails(Map<String, dynamic> culture) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      isScrollControlled: true,
      builder: (BuildContext context) {
        return DraggableScrollableSheet(
          initialChildSize: 0.75,
          maxChildSize: 0.95,
          minChildSize: 0.4,
          expand: false,
          builder: (_, controller) {
            return Padding(
              padding: const EdgeInsets.all(20),
              child: ListView(
                controller: controller,
                children: [
                  Center(
                    child: CircleAvatar(
                      radius: 50,
                      backgroundImage: AssetImage(culture['image']),
                    ),
                  ),
                  const SizedBox(height: 16),
                  Text(culture['nom'],
                      style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Color(0xFF15803D))),
                  const SizedBox(height: 16),
                  _buildSection("🌾 Description générale", culture['description']),
                  _buildSection("🌱 Physionomie", culture['physionomie']),
                  _buildSection("🌤️ Conditions idéales", culture['conditions']),
                  _buildSection("🦠 Maladies fréquentes", culture['maladies']),
                  _buildSection("🔍 Méthodes de détection", culture['tests']),
                  _buildSection("🧠 Conseils pratiques", culture['conseils']),
                ],
              ),
            );
          },
        );
      },
    );
  }


  Widget _buildSection(String title, String content) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: Colors.black87)),
          const SizedBox(height: 4),
          Text(content, style: const TextStyle(fontSize: 14, color: Colors.black87)),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        backgroundColor: Colors.white,
        body: Stack(
          children: [
            SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
    Padding(
                    padding: const EdgeInsets.all(16),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text('TipTiga',
                            style: TextStyle(
                                fontSize: 22,
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF15803D))),
                        IconButton(
                          icon: const Icon(Icons.camera_alt_outlined, color: Color(0xFF15803D)),
                          onPressed: _takePictureAndNavigate,
                        ),
                      ],
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.only(left: 16),
                    child: SizedBox(
                      height: 100,
                      child: ListView.builder(
                        scrollDirection: Axis.horizontal,
                        itemCount: cultures.length,
                        itemBuilder: (context, index) {
                          return GestureDetector(
                            onTap: () => _showCultureDetails(cultures[index]),
                            child: Container(
                              margin: const EdgeInsets.only(right: 12),
                              child: Column(
                                children: [
                                  CircleAvatar(
                                    radius: 30,
                                    backgroundImage: AssetImage(cultures[index]['image']!),
                                  ),
                                  const SizedBox(height: 6),
                                  Text(cultures[index]['nom']!, style: const TextStyle(fontWeight: FontWeight.w600)),
                                ],
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),

                  Stack(
                    alignment: Alignment.center,
                    children: [
                      CarouselSlider(
                        options: CarouselOptions(height: 180, autoPlay: true, viewportFraction: 1.0),
                        items: maladies.map((maladie) {
                          return Image.asset(
                            maladie['image']!,
                            fit: BoxFit.cover,
                            width: double.infinity,
                          );
                        }).toList(),
                      ),
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.9),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        margin: const EdgeInsets.symmetric(horizontal: 16),
                        child: Column(
                          children: [
                            const Text('Comment ça marche ?',
                                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                            const SizedBox(height: 12),
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                              children: const [
                                Column(
                                  children: [
                                    Icon(Icons.add_a_photo_outlined, color: Color(0xFF15803D)),
                                    Text('Photo'),
                                  ],
                                ),
                                Icon(Icons.arrow_forward_ios, size: 14, color: Colors.grey),
                                Column(
                                  children: [
                                    Icon(Icons.science_outlined, color: Colors.blue),
                                    Text('Analyse'),
                                  ],
                                ),
                                Icon(Icons.arrow_forward_ios, size: 14, color: Colors.grey),
                                Column(
                                  children: [
                                    Icon(Icons.health_and_safety_outlined, color: Colors.red),
                                    Text('Traitement'),
                                  ],
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 16),
                  Center(
                    child: ElevatedButton(
                      onPressed: _takePictureAndNavigate,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF15803D),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
                        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
                      ),
                      child: const Text('Prendre une photo',
                          style: TextStyle(fontSize: 16, color: Colors.white, fontWeight: FontWeight.bold)),
                    ),
                  ),

                  const SizedBox(height: 24),

                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('Météo', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
                        const SizedBox(height: 8),
                        _weather != null
                            ? Card(
                                elevation: 4,
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                                child: Padding(
                                  padding: const EdgeInsets.all(16),
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        '${_weather!.city}, ${_weather!.country}',
                                        style: const TextStyle(
                                            fontSize: 18,
                                            fontWeight: FontWeight.bold,
                                            color: Color(0xFF15803D)),
                                      ),
                                      const SizedBox(height: 8),
                                      Row(
                                        children: [
                                          const Icon(Icons.cloud_outlined, size: 48, color: Color(0xFF15803D)),
                                          const SizedBox(width: 16),
                                          Column(
                                            crossAxisAlignment: CrossAxisAlignment.start,
                                            children: [
                                              Text(
                                                '${_weather!.temperature.toStringAsFixed(1)}°C',
                                                style: const TextStyle(
                                                  fontSize: 32,
                                                  fontWeight: FontWeight.bold,
                                                  color: Color(0xFF15803D),
                                                ),
                                              ),
                                              Text(
                                                'Ressenti ${_weather!.feelsLike.toStringAsFixed(1)}°C. ${_weather!.description}.',
                                                style: const TextStyle(fontSize: 14, color: Colors.grey),
                                              ),
                                            ],
                                          ),
                                        ],
                                      ),
                                      const SizedBox(height: 12),
                                      Row(
                                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                        children: [
                                          _weatherInfoTile(Icons.air, '${_weather!.windSpeed} m/s', 'Vent'),
                                          _weatherInfoTile(Icons.opacity, '${_weather!.humidity}%', 'Humidité'),
                                          _weatherInfoTile(Icons.visibility, '${(_weather!.visibility / 1000).toStringAsFixed(1)} km', 'Visibilité'),
                                        ],
                                      ),
                                    ],
                                  ),
                                ),
                            )
                            : _weatherError
                                ? Column(
                                    children: [
                                      Container(
                                        padding: const EdgeInsets.all(16),
                                        decoration: BoxDecoration(
                                          color: Colors.red.shade50,
                                          border: Border.all(color: Colors.red.shade100),
                                          borderRadius: BorderRadius.circular(16),
                                        ),
                                        child: Row(
                                          children: const [
                                            Icon(Icons.wifi_off, color: Colors.red),
                                            SizedBox(width: 12),
                                            Expanded(
                                              child: Text(
                                                'Impossible de charger la météo. Vérifiez votre connexion internet ou activez la localisation.',
                                                style: TextStyle(color: Colors.red, fontWeight: FontWeight.w500),
                                              ),
                                            )
                                          ],
                                        ),
                                      ),
                                      const SizedBox(height: 8),
                                      Row(
                                        mainAxisAlignment: MainAxisAlignment.center,
                                        children: [
                                          ElevatedButton.icon(
                                            onPressed: () => _loadWeather(),
                                            icon: const Icon(Icons.refresh),
                                            label: const Text("Réessayer"),
                                            style: ElevatedButton.styleFrom(backgroundColor: Color(0xFF15803D)),
                                          ),
                                        ],
                                      )
                                    ],
                                  )
                                : const Center(child: CircularProgressIndicator()),
                      ],
                    ),
                  ),
                  const SizedBox(height: 32),
                ],
              ),
            ),
            if (_loading)
              Container(
                color: Colors.black.withOpacity(0.3),
                child: const Center(
                  child: CircularProgressIndicator(color: Color(0xFF15803D)),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
