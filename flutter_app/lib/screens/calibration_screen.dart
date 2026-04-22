import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import '../providers/app_state.dart';

class CalibrationScreen extends StatefulWidget {
  const CalibrationScreen({super.key});

  @override
  State<CalibrationScreen> createState() => _CalibrationScreenState();
}

class _CalibrationScreenState extends State<CalibrationScreen> {
  final _picker = ImagePicker();
  bool _uploading = false;

  Future<void> _captureAndUpload(AppState state) async {
    XFile? xfile;
    try {
      if (kIsWeb) {
        xfile = await _picker.pickImage(source: ImageSource.gallery);
      } else {
        xfile = await _picker.pickImage(
          source: ImageSource.camera,
          imageQuality: 90,
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Błąd aparatu: $e')),
        );
      }
      return;
    }

    if (xfile == null) return;

    setState(() => _uploading = true);
    final bytes = await xfile.readAsBytes();
    final ok = await state.uploadCalibImage(bytes);
    if (mounted) {
      setState(() => _uploading = false);
      if (ok) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Klatka przesłana'),
            backgroundColor: Colors.green,
          ),
        );
      }
    }
  }

  Future<void> _startCalibration(AppState state) async {
    await state.startCalibration();
    if (mounted && state.error == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Kalibracja uruchomiona — czekaj na wynik')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final theme = Theme.of(context);
    final session = state.session;
    final myFrames = session?.devices
            .where((d) => d.deviceId == state.deviceId)
            .firstOrNull
            ?.calibFrameCount ??
        0;
    final minFrames = session?.minCalibFrames ?? 0;
    final canCalibrate = state.isLeader && minFrames >= 3;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Kalibracja'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => state.refreshSession(),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Error banner
          if (state.error != null)
            Card(
              color: theme.colorScheme.errorContainer,
              margin: const EdgeInsets.only(bottom: 12),
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                child: Row(
                  children: [
                    Icon(Icons.error_outline, color: theme.colorScheme.error),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(state.error!,
                          style: TextStyle(color: theme.colorScheme.onErrorContainer)),
                    ),
                    IconButton(icon: const Icon(Icons.close), onPressed: state.clearError),
                  ],
                ),
              ),
            ),

          // Info banner
          if (state.info != null)
            Card(
              color: theme.colorScheme.primaryContainer,
              margin: const EdgeInsets.only(bottom: 12),
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                child: Row(
                  children: [
                    Icon(Icons.check_circle, color: theme.colorScheme.primary),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(state.info!,
                          style: TextStyle(color: theme.colorScheme.onPrimaryContainer)),
                    ),
                    IconButton(icon: const Icon(Icons.close), onPressed: state.clearInfo),
                  ],
                ),
              ),
            ),

          // Instructions
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.info_outline),
                      const SizedBox(width: 8),
                      Text('Instrukcja', style: theme.textTheme.titleMedium),
                    ],
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    '1. Przygotuj szachownicę kalibracyjną.\n'
                    '2. Sfotografuj ją z różnych kątów (min. 3 zdjęcia).\n'
                    '3. Gdy wszystkie urządzenia mają ≥ 3 klatki, lider uruchamia kalibrację.\n'
                    '4. Poczekaj na wynik — błąd reprojekcji < 1 px to dobry wynik.',
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 12),

          // Progress
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Postęp kalibracji', style: theme.textTheme.titleMedium),
                  const SizedBox(height: 12),
                  if (session != null)
                    ...session.devices.map(
                      (d) => Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Icon(
                                d.isLeader ? Icons.star : Icons.phone_android,
                                size: 16,
                                color: d.isLeader ? Colors.amber : theme.colorScheme.primary,
                              ),
                              const SizedBox(width: 6),
                              Text(d.deviceId),
                              const Spacer(),
                              Text('${d.calibFrameCount} klatek'),
                            ],
                          ),
                          const SizedBox(height: 4),
                          ClipRRect(
                            borderRadius: BorderRadius.circular(4),
                            child: LinearProgressIndicator(
                              value: (d.calibFrameCount / 10.0).clamp(0.0, 1.0),
                              minHeight: 8,
                            ),
                          ),
                          const SizedBox(height: 10),
                        ],
                      ),
                    ),
                  Text(
                    'Min. klatek (wszystkie urządzenia): $minFrames',
                    style: theme.textTheme.bodySmall,
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 12),

          // Capture button
          ElevatedButton.icon(
            onPressed: (_uploading || state.isLoading)
                ? null
                : () => _captureAndUpload(state),
            icon: _uploading
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : Icon(kIsWeb ? Icons.upload_file : Icons.camera_alt),
            label: Text(
              kIsWeb
                  ? 'Wybierz obraz kalibracyjny'
                  : 'Zrób zdjęcie szachownicy (${myFrames} / 10)',
            ),
          ),

          const SizedBox(height: 8),

          // Calibrate button (leader only)
          if (state.isLeader)
            ElevatedButton.icon(
              onPressed: (canCalibrate && !state.isLoading)
                  ? () => _startCalibration(state)
                  : null,
              icon: state.isLoading
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.calculate),
              label: Text(
                canCalibrate
                    ? 'Uruchom kalibrację'
                    : 'Potrzeba ≥ 3 klatek na urządzenie',
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: canCalibrate ? Colors.green : null,
                foregroundColor: canCalibrate ? Colors.white : null,
              ),
            )
          else
            const Card(
              child: Padding(
                padding: EdgeInsets.all(12),
                child: Row(
                  children: [
                    Icon(Icons.info_outline, size: 18),
                    SizedBox(width: 8),
                    Text('Kalibrację uruchamia lider sesji'),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
