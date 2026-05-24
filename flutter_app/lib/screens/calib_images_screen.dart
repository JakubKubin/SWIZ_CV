import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/models.dart';
import '../providers/app_state.dart';
import '../services/api_service.dart';
import '../theme/app_theme.dart';
import 'image_viewer_screen.dart';

class CalibImagesScreen extends StatefulWidget {
  const CalibImagesScreen({super.key});

  @override
  State<CalibImagesScreen> createState() => _CalibImagesScreenState();
}

class _CalibImagesScreenState extends State<CalibImagesScreen> {
  bool _loading = true;

  /// frame indices that appear in at least one camera device, sorted
  List<int> _pairIndices = [];

  /// frames per device: device_id → sorted list of FrameInfo
  Map<String, List<FrameInfo>> _framesByDevice = {};

  /// image bytes cache: "$deviceId/$frameIndex" → bytes
  final Map<String, Uint8List> _cache = {};

  late final AppState _appState;
  late final ApiService _api;

  @override
  void initState() {
    super.initState();
    _appState = context.read<AppState>();
    _api = ApiService(_appState.serverUrl);
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    final session = _appState.session;
    if (session == null) {
      setState(() => _loading = false);
      return;
    }

    final cameras = session.devices.where((d) => d.isCamera).toList();
    final results = await Future.wait(
      cameras.map((d) => _api
          .listCalibImages(session.sessionId, d.deviceId)
          .then((frames) => MapEntry(d.deviceId, frames))
          .catchError((_) => MapEntry(d.deviceId, <FrameInfo>[]))),
    );

    final byDevice = Map.fromEntries(results);
    final allIndices = <int>{};
    for (final frames in byDevice.values) {
      allIndices.addAll(frames.map((f) => f.index));
    }

    setState(() {
      _framesByDevice = byDevice;
      _pairIndices = allIndices.toList()..sort();
      _loading = false;
    });
  }

  Future<Uint8List?> _loadImage(String deviceId, int frameIndex) async {
    final key = '$deviceId/$frameIndex';
    if (_cache.containsKey(key)) return _cache[key];
    final session = _appState.session;
    if (session == null) return null;
    try {
      final url =
          '${_appState.serverUrl}/sessions/${session.sessionId}/calibration/images/$deviceId/$frameIndex';
      final bytes = await _api.getImageBytes(url);
      _cache[key] = bytes;
      return bytes;
    } catch (_) {
      return null;
    }
  }

  void _showViewer(int pairListIndex, int cameraIndex) {
    final cameras =
        _appState.session?.devices.where((d) => d.isCamera).toList() ?? [];
    final images = <ImageEntry>[];
    for (var pi = 0; pi < _pairIndices.length; pi++) {
      final pairIdx = _pairIndices[pi];
      for (var ci = 0; ci < cameras.length; ci++) {
        images.add(ImageEntry(
          deviceId: cameras[ci].deviceId,
          frameIndex: pairIdx,
          label: '#$pairIdx · ${ci == 0 ? 'L' : 'R'}',
        ));
      }
    }
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ImageViewerScreen(
          images: images,
          initialIndex: pairListIndex * cameras.length + cameraIndex,
          loadImage: _loadImage,
        ),
      ),
    );
  }

  Future<void> _deletePair(int frameIndex) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Usuń parę kalibracyjną'),
        content:
            Text('Usunąć klatkę #$frameIndex ze wszystkich urządzeń?\nTej operacji nie można cofnąć.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Anuluj')),
          TextButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Usuń')),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;

    final ok = await _appState.deleteCalibPair(frameIndex);
    if (!mounted) return;
    if (ok) {
      // Remove frame from local state and purge cache
      setState(() {
        _pairIndices.remove(frameIndex);
        for (final deviceId in _framesByDevice.keys) {
          _framesByDevice[deviceId]
              ?.removeWhere((f) => f.index == frameIndex);
          _cache.remove('$deviceId/$frameIndex');
        }
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final tt = theme.textTheme;
    final session = state.session;

    final cameras = session?.devices.where((d) => d.isCamera).toList() ?? [];

    return Scaffold(
      appBar: AppBar(
        title: const Text('Zdjęcia kalibracyjne'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_outlined),
            tooltip: 'Odśwież',
            onPressed: _load,
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _pairIndices.isEmpty
              ? Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.photo_library_outlined,
                          size: 48, color: cs.onSurfaceVariant),
                      const SizedBox(height: 12),
                      Text('Brak zdjęć kalibracyjnych',
                          style: tt.bodyMedium
                              ?.copyWith(color: cs.onSurfaceVariant)),
                    ],
                  ),
                )
              : ListView.builder(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  itemCount: _pairIndices.length,
                  itemBuilder: (context, i) {
                    final idx = _pairIndices[i];
                    return _PairRow(
                      frameIndex: idx,
                      cameras: cameras,
                      loadImage: _loadImage,
                      isLeader: state.isLeader,
                      onDelete: () => _deletePair(idx),
                      onTap: (cameraIndex) => _showViewer(i, cameraIndex),
                      cs: cs,
                      tt: tt,
                    );
                  },
                ),
    );
  }
}

class _PairRow extends StatelessWidget {
  final int frameIndex;
  final List<DeviceInfo> cameras;
  final Future<Uint8List?> Function(String deviceId, int frameIndex) loadImage;
  final bool isLeader;
  final VoidCallback onDelete;
  final void Function(int cameraIndex) onTap;
  final ColorScheme cs;
  final TextTheme tt;

  const _PairRow({
    required this.frameIndex,
    required this.cameras,
    required this.loadImage,
    required this.isLeader,
    required this.onDelete,
    required this.onTap,
    required this.cs,
    required this.tt,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 4),
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: Row(
          children: [
            // Frame index badge
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                color: cs.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(6),
              ),
              alignment: Alignment.center,
              child: Text(
                '#$frameIndex',
                style: tt.labelSmall?.copyWith(
                    color: cs.onSurfaceVariant, fontWeight: FontWeight.w600),
              ),
            ),
            const SizedBox(width: 10),
            // Thumbnails for each camera device — index 0 = L, 1 = R
            ...cameras.asMap().entries.map((entry) {
              final ci = entry.key;
              final dev = entry.value;
              final label = ci == 0 ? 'L' : 'R';
              return Padding(
                padding: const EdgeInsets.only(right: 6),
                child: Column(
                  children: [
                    _Thumbnail(
                      future: loadImage(dev.deviceId, frameIndex),
                      size: 72,
                      onTap: () => onTap(ci),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      label,
                      style: tt.labelSmall?.copyWith(
                          color: ci == 0
                              ? AppColors.stateReady
                              : cs.onSurfaceVariant),
                    ),
                  ],
                ),
              );
            }),
            const Spacer(),
            if (isLeader)
              IconButton(
                icon: Icon(Icons.delete_outline, color: cs.error, size: 20),
                tooltip: 'Usuń parę',
                onPressed: onDelete,
              ),
          ],
        ),
      ),
    );
  }
}

class _Thumbnail extends StatefulWidget {
  final Future<Uint8List?> future;
  final double size;
  final VoidCallback? onTap;

  const _Thumbnail({required this.future, required this.size, this.onTap});

  @override
  State<_Thumbnail> createState() => _ThumbnailState();
}

class _ThumbnailState extends State<_Thumbnail> {
  late Future<Uint8List?> _future;

  @override
  void initState() {
    super.initState();
    _future = widget.future;
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return GestureDetector(
      onTap: widget.onTap,
      child: SizedBox(
        width: widget.size,
        height: widget.size,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: FutureBuilder<Uint8List?>(
            future: _future,
            builder: (context, snap) {
              if (snap.connectionState == ConnectionState.waiting) {
                return Container(
                  color: cs.surfaceContainerHighest,
                  child: const Center(
                      child: CircularProgressIndicator(strokeWidth: 2)),
                );
              }
              if (snap.hasData && snap.data != null) {
                return Image.memory(snap.data!, fit: BoxFit.cover);
              }
              return Container(
                color: cs.surfaceContainerHighest,
                child: Icon(Icons.broken_image_outlined,
                    size: 24, color: cs.onSurfaceVariant),
              );
            },
          ),
        ),
      ),
    );
  }
}
