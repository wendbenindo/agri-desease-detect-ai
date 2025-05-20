// // File: lib/pages/home_page.dart
// import 'package:flutter/material.dart';
// import 'package:intl/intl.dart';

// class HomePage extends StatelessWidget {
//   const HomePage({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return SafeArea(
//       child: Column(
//         crossAxisAlignment: CrossAxisAlignment.start,
//         children: [
//           Padding(
//             padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
//             child: Text(
//               'TipTiga',
//               style: Theme.of(context).textTheme.displayLarge,
//             ),
//           ),
//           Expanded(
//             child: SingleChildScrollView(
//               child: Column(
//                 children: [
//                   Padding(
//                     padding: const EdgeInsets.symmetric(horizontal: 16),
//                     child: Image.asset(
//                       'assets/images/home_banner.png',
//                       height: 180,
//                       fit: BoxFit.contain,
//                     ),
//                   ),
//                   const SizedBox(height: 16),
//                   Padding(
//                     padding: const EdgeInsets.symmetric(horizontal: 16),
//                     child: Card(
//                       child: Padding(
//                         padding: const EdgeInsets.all(16),
//                         child: Column(
//                           crossAxisAlignment: CrossAxisAlignment.start,
//                           children: const [
//                             Text(
//                               'Comment ça marche ?',
//                               style: TextStyle(
//                                 fontSize: 18,
//                                 fontWeight: FontWeight.bold,
//                               ),
//                             ),
//                             SizedBox(height: 8),
//                             Text('1. Prendre une photo de la plante.'),
//                             Text('2. Scanner avec notre IA locale.'),
//                             Text('3. Obtenir un diagnostic immédiat.'),
//                           ],
//                         ),
//                       ),
//                     ),
//                   ),
//                   const SizedBox(height: 16),
//                   ElevatedButton.icon(
//                     onPressed: () {
//                       // Action pour prendre une photo
//                     },
//                     icon: const Icon(Icons.camera_alt),
//                     label: const Text('Prendre une photo'),
//                   ),
//                   const SizedBox(height: 32),
//                   Padding(
//                     padding: const EdgeInsets.symmetric(horizontal: 16),
//                     child: Card(
//                       color: const Color(0xFF3B82F6), // Tailwind blue-500
//                       child: Padding(
//                         padding: const EdgeInsets.all(16),
//                         child: Column(
//                           crossAxisAlignment: CrossAxisAlignment.start,
//                           children: [
//                             Row(
//                               children: [
//                                 const Icon(Icons.thermostat, color: Colors.white),
//                                 const SizedBox(width: 8),
//                                 Text('Température: 32°C', style: Theme.of(context).textTheme.bodyLarge!.copyWith(color: Colors.white)),
//                               ],
//                             ),
//                             const SizedBox(height: 8),
//                             Row(
//                               children: [
//                                 const Icon(Icons.air, color: Colors.white),
//                                 const SizedBox(width: 8),
//                                 Text('Vent: 15 km/h', style: Theme.of(context).textTheme.bodyLarge!.copyWith(color: Colors.white)),
//                               ],
//                             ),
//                             const SizedBox(height: 8),
//                             Row(
//                               children: [
//                                 const Icon(Icons.water_drop, color: Colors.white),
//                                 const SizedBox(width: 8),
//                                 Text('Humidité: 64%', style: Theme.of(context).textTheme.bodyLarge!.copyWith(color: Colors.white)),
//                               ],
//                             ),
//                           ],
//                         ),
//                       ),
//                     ),
//                   ),
//                   const SizedBox(height: 16),
//                 ],
//               ),
//             ),
//           ),
//         ],
//       ),
//     );
//   }
// }



// import 'package:flutter/material.dart';
// import 'package:intl/intl.dart';

// class HomePage extends StatelessWidget {
//   const HomePage({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return SafeArea(
//       child: Column(
//         crossAxisAlignment: CrossAxisAlignment.start,
//         children: [
//           Padding(
//             padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
//             child: Text(
//               'TipTiga',
//               style: Theme.of(context).textTheme.displayLarge,
//             ),
//           ),
//           Expanded(
//             child: SingleChildScrollView(
//               child: Column(
//                 children: [
//                   Padding(
//                     padding: const EdgeInsets.symmetric(horizontal: 16),
//                     child: Image.asset(
//                       'assets/images/home_banner.png',
//                       height: 160,
//                       fit: BoxFit.contain,
//                     ),
//                   ),
//                   const SizedBox(height: 16),
//                   Padding(
//                     padding: const EdgeInsets.symmetric(horizontal: 16),
//                     child: Card(
//                       elevation: 4,
//                       shape: RoundedRectangleBorder(
//                         borderRadius: BorderRadius.circular(16),
//                       ),
//                       child: Padding(
//                         padding: const EdgeInsets.all(16),
//                         child: Column(
//                           crossAxisAlignment: CrossAxisAlignment.center,
//                           children: [
//                             const Text(
//                               'Heal your crop',
//                               style: TextStyle(
//                                 fontSize: 18,
//                                 fontWeight: FontWeight.bold,
//                               ),
//                             ),
//                             const SizedBox(height: 12),
//                             Row(
//                               mainAxisAlignment: MainAxisAlignment.spaceEvenly,
//                               children: const [
//                                 Column(
//                                   children: [
//                                     Icon(Icons.camera_alt, size: 30, color: Colors.green),
//                                     SizedBox(height: 4),
//                                     Text('Take a picture'),
//                                   ],
//                                 ),
//                                 Icon(Icons.arrow_forward_ios, size: 16, color: Colors.grey),
//                                 Column(
//                                   children: [
//                                     Icon(Icons.insights, size: 30, color: Colors.blue),
//                                     SizedBox(height: 4),
//                                     Text('Get result'),
//                                   ],
//                                 ),
//                                 Icon(Icons.arrow_forward_ios, size: 16, color: Colors.grey),
//                                 Column(
//                                   children: [
//                                     Icon(Icons.local_pharmacy, size: 30, color: Colors.redAccent),
//                                     SizedBox(height: 4),
//                                     Text('Diagnosis'),
//                                   ],
//                                 ),
//                               ],
//                             ),
//                             const SizedBox(height: 16),
//                             Row(
//                               mainAxisAlignment: MainAxisAlignment.spaceEvenly,
//                               children: [
//                                 ElevatedButton.icon(
//                                   onPressed: () {
//                                     // Capture photo avec la caméra
//                                   },
//                                   icon: const Icon(Icons.photo_camera),
//                                   label: const Text('Take a picture'),
//                                 ),
//                                 ElevatedButton.icon(
//                                   onPressed: () {
//                                     // Sélectionner une image depuis la galerie
//                                   },
//                                   icon: const Icon(Icons.image),
//                                   label: const Text('From gallery'),
//                                   style: ElevatedButton.styleFrom(
//                                     backgroundColor: Colors.white,
//                                     foregroundColor: Colors.green,
//                                     side: const BorderSide(color: Colors.green),
//                                   ),
//                                 ),
//                               ],
//                             ),
//                           ],
//                         ),
//                       ),
//                     ),
//                   ),
//                   const SizedBox(height: 24),
//                   Padding(
//                     padding: const EdgeInsets.symmetric(horizontal: 16),
//                     child: Column(
//                       crossAxisAlignment: CrossAxisAlignment.start,
//                       children: [
//                         const Text(
//                           'Weather',
//                           style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
//                         ),
//                         const SizedBox(height: 8),
//                         Card(
//                           shape: RoundedRectangleBorder(
//                             borderRadius: BorderRadius.circular(16),
//                           ),
//                           color: const Color(0xFF3B82F6),
//                           child: Padding(
//                             padding: const EdgeInsets.all(16),
//                             child: Row(
//                               mainAxisAlignment: MainAxisAlignment.spaceBetween,
//                               children: [
//                                 Column(
//                                   crossAxisAlignment: CrossAxisAlignment.start,
//                                   children: [
//                                     Text('58.1 °C', style: Theme.of(context).textTheme.headlineMedium!.copyWith(color: Colors.white)),
//                                     const Text('overcast clouds', style: TextStyle(color: Colors.white70)),
//                                   ],
//                                 ),
//                                 const Icon(Icons.cloud, size: 40, color: Colors.white),
//                               ],
//                             ),
//                           ),
//                         ),
//                       ],
//                     ),
//                   ),
//                   const SizedBox(height: 16),
//                 ],
//               ),
//             ),
//           ),
//         ],
//       ),
//     );
//   }
// }


// File: lib/pages/home_page.dart
import 'package:flutter/material.dart';
import 'package:carousel_slider/carousel_slider.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    final List<Map<String, String>> maladies = [
      {
        'image': 'assets/images/mildiou.png',
        'nom': 'Mildiou du maïs'
      },
      {
        'image': 'assets/images/rouille.png',
        'nom': 'Rouille du sorgho'
      },
      {
        'image': 'assets/images/tache_feuille.png',
        'nom': 'Tache foliaire'
      },
    ];

    return SafeArea(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            child: Text(
              'TipTiga',
              style: Theme.of(context).textTheme.displayLarge,
            ),
          ),
          CarouselSlider(
            options: CarouselOptions(
              height: 160,
              autoPlay: true,
              enlargeCenterPage: true,
              viewportFraction: 0.85,
            ),
            items: maladies.map((maladie) {
              return Builder(
                builder: (BuildContext context) {
                  return Card(
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                    elevation: 4,
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Expanded(
                          child: ClipRRect(
                            borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
                            child: Image.asset(
                              maladie['image']!,
                              fit: BoxFit.cover,
                              width: double.infinity,
                            ),
                          ),
                        ),
                        Padding(
                          padding: const EdgeInsets.all(8.0),
                          child: Text(
                            maladie['nom']!,
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 16,
                            ),
                          ),
                        ),
                      ],
                    ),
                  );
                },
              );
            }).toList(),
          ),
          const SizedBox(height: 16),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Card(
              elevation: 4,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const Text(
                      'Comment ça marche ?',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: const [
                        Column(
                          children: [
                            Icon(Icons.camera, size: 30, color: Colors.green),
                            SizedBox(height: 4),
                            Text('Photo'),
                          ],
                        ),
                        Icon(Icons.arrow_forward_ios, size: 16, color: Colors.grey),
                        Column(
                          children: [
                            Icon(Icons.analytics, size: 30, color: Colors.blue),
                            SizedBox(height: 4),
                            Text('Analyse'),
                          ],
                        ),
                        Icon(Icons.arrow_forward_ios, size: 16, color: Colors.grey),
                        Column(
                          children: [
                            Icon(Icons.medical_information, size: 30, color: Colors.redAccent),
                            SizedBox(height: 4),
                            Text('Diagnostic'),
                          ],
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        ElevatedButton(
                          onPressed: () {
                            // Prendre photo
                          },
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                            backgroundColor: const Color(0xFF15803D),
                          ),
                          child: const Icon(Icons.camera_alt, color: Colors.white),
                        ),
                        ElevatedButton(
                          onPressed: () {
                            // Sélectionner depuis la galerie
                          },
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                            backgroundColor: Colors.white,
                            side: const BorderSide(color: Color(0xFF15803D)),
                          ),
                          child: const Icon(Icons.photo_library, color: Color(0xFF15803D)),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 24),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Météo',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 8),
                Card(
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  color: const Color(0xFF3B82F6),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('32°C', style: Theme.of(context).textTheme.headlineMedium!.copyWith(color: Colors.white)),
                            const Text('Ciel couvert', style: TextStyle(color: Colors.white70)),
                          ],
                        ),
                        const Icon(Icons.cloud, size: 40, color: Colors.white),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
        ],
      ),
    );
  }
}
