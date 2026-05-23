import 'dart:typed_data';

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
  bool _picking = false;

  Future<void> _pickFromGallery(AppState state) async {
    setState(() => _picking = true);
    try {
      final files = await _picker.pickMultiImage(imageQuality: 90);
      if (files.isEmpty) return;
      final loaded = <({Uint8List bytes, String name})>[];
      for (final f in files) {
        loaded.add((bytes: await f.readAsBytes(), name: f.name));
      }
      state.addPendingCalibImages(loaded);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Błąd: $e')));
      }
    } finally {
      if (mounted) setState(() => _picking = false);
    }
  }

  Future<void> _captureFromCamera(AppState state) async {
    setState(() => _picking = true);
    try {
      final file = await _picker.pickImage(
        source: ImageSource.camera,
        imageQuality: 90,
      );
      if (file == null) return;
      state.addPendingCalibImages([
        (bytes: await file.readAsBytes(), name: file.name),
      ]);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Błąd aparatu: $e')));
      }
    } finally {
      if (mounted) setState(() => _picking = false);
    }
  }

  Future<void> _uploadAll(AppState state) async {
    final ok = await state.uploadAllPendingCalibImages();
    if (mounted && ok) {
      final total = state.myDevice?.calibFrameCount ?? 0;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Przesłano — łącznie $total klatek')),
      );
    }
  }

  Future<void> _startCalibration(AppState state) async {
    await state.startCalibration();
    if (mounted && state.error == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text('Kalibracja uruchomiona — czekaj na wynik')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final tt = theme.textTheme;
    final session = state.session;

    final myFrames = state.myDevice?.calibFrameCount ?? 0;
    final minFrames = session?.minCalibFrames ?? 0;
    final canCalibrate = state.isLeader && minFrames >= 3;
    final pending = state.pendingCalibImages;
    final busy = _picking || state.isLoading;

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
          AppBanners(
            error: state.error,
            info: state.info,
            onClearError: state.clearError,
            onClearInfo: state.clearInfo,
          ),

          // Instructions
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Instrukcja',
                      style: tt.titleSmall?.copyWith(
                          color: cs.primary, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 8),
                  Text(
                    '1. Przygotuj szachownicę kalibracyjną.\n'
                    '2. Dodaj zdjęcia z różnych kątów (min. 3).\n'
                    '3. Usuń niepotrzebne, zatwierdź przesyłając wszystkie.\n'
                    '4. Gdy wszystkie urządzenia mają min. 3 klatki, lider uruchamia kalibrację.',
                    style: tt.bodySmall
                        ?.copyWith(color: cs.onSurfaceVariant, height: 1.5),
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
                  Text('Postęp kalibracji',
                      style: tt.titleSmall?.copyWith(
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
                                width: 6,
                                height: 6,
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
                                  style: tt.bodySmall
                                      ?.copyWith(color: cs.onSurfaceVariant)),
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

          // Pick buttons
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Dodaj obrazy',
                      style: tt.titleSmall?.copyWith(
                          color: cs.primary, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: busy ? null : () => _pickFromGallery(state),
                          icon: _picking
                              ? const SizedBox(
                                  width: 16,
                                  height: 16,
                                  child:
                                      CircularProgressIndicator(strokeWidth: 2))
                              : const Icon(Icons.photo_library_outlined),
                          label: const Text('Galeria'),
                        ),
                      ),
                      if (!kIsWeb) ...[
                        const SizedBox(width: 10),
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed:
                                busy ? null : () => _captureFromCamera(state),
                            icon: const Icon(Icons.camera_alt_outlined),
                            label: Text('Aparat ($myFrames)'),
                          ),
                        ),
                      ],
                    ],
                  ),
                ],
              ),
            ),
          ),

          // Staging area
          if (pending.isNotEmpty) ...[
            const SizedBox(height: 12),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text(
                          'Wybrane obrazy (${pending.length})',
                          style: tt.titleSmall?.copyWith(
                              color: cs.primary, fontWeight: FontWeight.w600),
                        ),
                        const Spacer(),
                        TextButton.icon(
                          onPressed:
                              busy ? null : () => state.clearPendingCalibImages(),
                          icon: const Icon(Icons.delete_sweep_outlined, size: 18),
                          label: const Text('Usuń wszystkie'),
                          style: TextButton.styleFrom(
                              foregroundColor: cs.error,
                              padding: EdgeInsets.zero),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    GridView.builder(
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      gridDelegate:
                          const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 4,
                        crossAxisSpacing: 6,
                        mainAxisSpacing: 6,
                      ),
                      itemCount: pending.length,
                      itemBuilder: (context, i) {
                        return _StagedImageTile(
                          bytes: pending[i].bytes,
                          name: pending[i].name,
                          index: i + 1,
                          onRemove: busy
                              ? null
                              : () => state.removePendingCalibImage(i),
                        );
                      },
                    ),
                    const SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        onPressed: busy ? null : () => _uploadAll(state),
                        icon: state.isLoading
                            ? const SizedBox(
                                width: 18,
                                height: 18,
                                child: CircularProgressIndicator(strokeWidth: 2))
                            : const Icon(Icons.cloud_upload_outlined),
                        label: Text(
                            'Wyślij ${pending.length} ${_framesLabel(pending.length)}'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: cs.primary,
                          foregroundColor: cs.onPrimary,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],

          const SizedBox(height: 12),

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
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                child: Row(
                  children: [
                    Icon(Icons.info_outline,
                        size: 16, color: cs.onSurfaceVariant),
                    const SizedBox(width: 8),
                    Text('Kalibrację uruchamia lider sesji',
                        style:
                            tt.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
                  ],
                ),
              ),
            ),

          const SizedBox(height: 8),
        ],
      ),
    );
  }

  String _framesLabel(int n) {
    if (n == 1) return 'obraz';
    if (n >= 2 && n <= 4) return 'obrazy';
    return 'obrazów';
  }
}

class _StagedImageTile extends StatelessWidget {
  final Uint8List bytes;
  final String name;
  final int index;
  final VoidCallback? onRemove;

  const _StagedImageTile({
    required this.bytes,
    required this.name,
    required this.index,
    required this.onRemove,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Stack(
      fit: StackFit.expand,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: Image.memory(bytes, fit: BoxFit.cover),
        ),
        // index badge
        Positioned(
          left: 4,
          bottom: 4,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
            decoration: BoxDecoration(
              color: Colors.black54,
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              '$index',
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.w600),
            ),
          ),
        ),
        // remove button
        Positioned(
          top: 2,
          right: 2,
          child: GestureDetector(
            onTap: onRemove,
            child: Container(
              width: 20,
              height: 20,
              decoration: BoxDecoration(
                color: onRemove != null ? cs.error : cs.outline,
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.close, size: 13, color: Colors.white),
            ),
          ),
        ),
      ],
    );
  }
}
