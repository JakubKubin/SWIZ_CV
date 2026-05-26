import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'providers/app_state.dart';
import 'screens/home_screen.dart';
import 'theme/app_theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final prefs = await SharedPreferences.getInstance();
  final deviceName = await resolveDeviceName();
  runApp(
    ChangeNotifierProvider(
      create: (_) => AppState(prefs, deviceName: deviceName),
      child: const StereoVisionApp(),
    ),
  );
}

/// Returns a sanitized device name string, e.g. "samsung_galaxy_s21" or "jakubs_iphone".
/// Returns null if unavailable (web or plugin failure).
Future<String?> resolveDeviceName() async {
  if (kIsWeb) return null;
  try {
    final plugin = DeviceInfoPlugin();
    String? raw;
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        final info = await plugin.androidInfo;
        raw = info.model; // e.g. "Pixel 7", "SM-G998B"
      case TargetPlatform.iOS:
        final info = await plugin.iosInfo;
        raw = info.model; // e.g. "iPhone", "iPad Pro"
      case TargetPlatform.macOS:
        final info = await plugin.macOsInfo;
        raw = info.computerName;
      case TargetPlatform.windows:
        final info = await plugin.windowsInfo;
        raw = info.computerName;
      default:
        return null;
    }
    if (raw.trim().isEmpty) return null;
    return _sanitize(raw);
  } catch (_) {
    return null;
  }
}

/// "Samsung Galaxy S21" → "samsung_galaxy_s21"
/// "Jakub's iPhone"    → "jakubs_iphone"
String _sanitize(String s) => s
    .toLowerCase()
    .replaceAll(RegExp(r"[^a-z0-9]+"), '_')
    .replaceAll(RegExp(r'_+'), '_')
    .replaceAll(RegExp(r'^_+|_+$'), '');

class StereoVisionApp extends StatelessWidget {
  const StereoVisionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'StereoVision',
      debugShowCheckedModeBanner: false,
      theme: buildTheme(),
      home: const HomeScreen(),
    );
  }
}
