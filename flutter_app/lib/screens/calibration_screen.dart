import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import '../providers/app_state.dart';
import '../theme/app_theme.dart';
import '../widgets/app_banner.dart';

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
          const SnackBar(content: Text('Klatka przesłana')),
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
    final state   = context.watch<AppState>();
    final theme   = Theme.of(context);
    final cs      = theme.colorScheme;
    final tt      = theme.textTheme;
    final session = state.session;

    final myFrames = session?.devices
            .where((d) => d.deviceId == state.deviceId)
            .firstOrNull
            ?.calibFrameCount ??
        0;
    final minFrames    = session?.minCalibFrames ?? 0;
    final canCalibrate = state.isLeader && minFrames >= 3;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Kalibracja'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_outlined),
            tooltip: 'Odśwież',
            onPressed: () => state.refreshSession(),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        children: [
          if (state.error != null)
            AppBanner(
              color: cs.errorContainer,
              textColor: cs.onErrorContainer,
              icon: Icons.error_outline,
              message: state.error!,
              onClose: state.clearError,
            ),
          if (state.info != null)
            AppBanner(
              color: cs.secondaryContainer,
              textColor: cs.onSecondaryContainer,
              icon: Icons.info_outline,
              message: state.info!,
              onClose: state.clearInfo,
            ),

          // Instructions
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Instrukcja', style: tt.titleSmall?.copyWith(
                      color: cs.primary, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 8),
                  Text(
                    '1. Przygotuj szachownicę kalibracyjną.\n'
                    '2. Sfotografuj ją z różnych kątów (min. 3 zdjęcia).\n'
                    '3. Gdy wszystkie urządzenia mają min. 3 klatki, lider uruchamia kalibrację.\n'
                    '4. Poczekaj na wynik — błąd reprojekcji < 1 px to dobry wynik.',
                    style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant, height: 1.5),
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
                  Text('Postęp kalibracji', style: tt.titleSmall?.copyWith(
                      color: cs.primary, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 12),
                  if (session != null)
                    ...session.devices.map(
                      (d) => Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Container(
                                width: 6, height: 6,
                                decoration: BoxDecoration(
                                  color: d.isLeader
                                      ? AppColors.stateReady
                                      : cs.onSurfaceVariant,
                                  shape: BoxShape.circle,
                                ),
                              ),
                              const SizedBox(width: 8),
                              Text(d.deviceId, style: tt.bodySmall),
                              const Spacer(),
                              Text('${d.calibFrameCount} klatek',
                                  style: tt.bodySmall?.copyWith(
                                      color: cs.onSurfaceVariant)),
                            ],
                          ),
                          const SizedBox(height: 6),
                          ClipRRect(
                            borderRadius: BorderRadius.circular(4),
                            child: LinearProgressIndicator(
                              value: (d.calibFrameCount / 10.0).clamp(0.0, 1.0),
                              minHeight: 6,
                              backgroundColor: cs.surfaceContainerHighest,
                            ),
                          ),
                          const SizedBox(height: 10),
                        ],
                      ),
                    ),
                  Text(
                    'Minimum klatek (wszystkie urządzenia): $minFrames',
                    style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant),
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
                    width: 18, height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : Icon(kIsWeb ? Icons.upload_file_outlined : Icons.camera_alt_outlined),
            label: Text(
              kIsWeb
                  ? 'Wybierz obraz kalibracyjny'
                  : 'Zrób zdjęcie szachownicy ($myFrames / 10)',
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
                      width: 18, height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.calculate_outlined),
              label: Text(
                canCalibrate
                    ? 'Uruchom kalibrację'
                    : 'Potrzeba min. 3 klatek na urządzenie',
              ),
              style: canCalibrate
                  ? ElevatedButton.styleFrom(
                      backgroundColor: AppColors.stateDone,
                      foregroundColor: Colors.white,
                    )
                  : null,
            )
          else
            Card(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                child: Row(
                  children: [
                    Icon(Icons.info_outline, size: 16, color: cs.onSurfaceVariant),
                    const SizedBox(width: 8),
                    Text('Kalibrację uruchamia lider sesji',
                        style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
                  ],
                ),
              ),
            ),

          const SizedBox(height: 8),
        ],
      ),
    );
  }
}
