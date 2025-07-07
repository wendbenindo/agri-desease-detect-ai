import 'dart:io' show Platform;
import 'package:agri_desease_detect_app/pages/communitypage.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:geolocator/geolocator.dart';

import 'package:agri_desease_detect_app/widgets/theme.dart';
import 'package:agri_desease_detect_app/widgets/splashscreen.dart';
import 'package:agri_desease_detect_app/pages/homepage.dart';
import 'package:agri_desease_detect_app/pages/diagnosticpage.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    if (!kIsWeb) {
      await dotenv.load(fileName: ".env");
    } else {
      await dotenv.load(fileName: "assets/.env");
    }

    await _handleLocationPermission(); // Demande de permission
  } catch (e) {
    debugPrint("Erreur de chargement ou permissions : $e");
  }

  runApp(const TipTigaApp());
}

Future<void> _handleLocationPermission() async {
  bool serviceEnabled;
  LocationPermission permission;

  serviceEnabled = await Geolocator.isLocationServiceEnabled();
  if (!serviceEnabled) {
    debugPrint('Les services de localisation sont désactivés.');
    return;
  }

  permission = await Geolocator.checkPermission();
  if (permission == LocationPermission.denied) {
    permission = await Geolocator.requestPermission();
    if (permission == LocationPermission.denied) {
      debugPrint('Permission de localisation refusée.');
      return;
    }
  }

  if (permission == LocationPermission.deniedForever) {
    debugPrint('Permission refusée définitivement. Veuillez l’activer dans les paramètres.');
    return;
  }

  debugPrint('Permission de localisation accordée.');
}

class TipTigaApp extends StatelessWidget {
  const TipTigaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TipTiga',
      theme: tipTigaTheme,
      debugShowCheckedModeBanner: false,
      home: const SplashScreen(),
    );
  }
}

class NavigationController extends StatefulWidget {
  const NavigationController({super.key});

  @override
  State<NavigationController> createState() => _NavigationControllerState();
}

class _NavigationControllerState extends State<NavigationController> {
  int _selectedIndex = 0;

  final List<Widget> _pages = const [
    HomePage(),
    DiagnosticPage(),
    CommunityPage(), // tu peux la réactiver plus tard
  ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        backgroundColor: Colors.white,
        selectedItemColor: Colors.green.shade700,
        unselectedItemColor: Colors.grey,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: 'Accueil',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.health_and_safety),
            label: 'Diagnostic',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.groups),
            label: 'Communauté',
          ),
        ],
      ),
    );
  }
}
