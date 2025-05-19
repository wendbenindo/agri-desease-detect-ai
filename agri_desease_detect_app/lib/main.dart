// File: lib/main.dart
import 'package:flutter/material.dart';
import 'package:agri_desease_detect_app/widgets/theme.dart';
import 'package:agri_desease_detect_app/widgets/splashscreen.dart';
import 'package:agri_desease_detect_app/pages/homepage.dart';


void main() {
  runApp(const TipTigaApp());
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
    // DetectionPage(),
    // CommunityPage(),
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
            icon: Icon(Icons.search),
            label: 'Détection',
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
