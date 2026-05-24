import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/models.dart';
import '../providers/app_state.dart';
import '../services/api_service.dart';
import 'image_viewer_screen.dart';

class CaptureImagesScreen extends StatefulWidget {
  const CaptureImagesScreen({super.key});

  @override
  State<CaptureImagesScreen> createState() => _CaptureImagesScreenState();
}

class _CaptureImagesScreenState extends State<CaptureImagesScreen> {
  bool _loading = true;

  /// captures per device: device_id → list of FrameInfo (mutable for removal)
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
          .listCaptureImages(session.sessionId, d.deviceId)
          .then((frames) => MapEntry(d.deviceId, frames))
          .catchError((_) => MapEntry(d.deviceId, <FrameInfo>[]))),
    );

    setState(() {
      _framesByDevice = Map.fromEntries(results);
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
          '${_appState.serverUrl}/sessions/${session.sessionId}/capture/images/$deviceId/$frameIndex';
      final bytes = await _api.getImageBytes(url);
      _cache[key] = bytes;
      return bytes;
    } catch (_) {
      return null;
    }
  }

  void _showViewer(String deviceId, int frameIndex) {
    final cameras =
        _appState.session?.devices.where((d) => d.isCamera).toList() ?? [];
    final images = <ImageEntry>[];
    int startIndex = 0;
    bool found = false;
    for (final dev in cameras) {
      for (final frame in (_framesByDevice[dev.deviceId] ?? [])) {
        if (!found) {
          if (dev.deviceId == deviceId && frame.index == frameIndex) {
            found = true;
          } else {
            startIndex++;
          }
        }
        images.add(ImageEntry(
          deviceId: dev.deviceId,
          frameIndex: frame.index,
          label: '${dev.deviceId} · #${frame.index}',
        ));
      }
    }
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ImageViewerScreen(
          images: images,
          initialIndex: startIndex,
          loadImage: _loadImage,
        ),
      ),
    );
  }

  Future<void> _deleteFrame(String deviceId, int frameIndex) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Usuń zdjęcie'),
        content: Text(
            'Usunąć zdjęcie #$frameIndex urządzenia $deviceId?\nTej operacji nie można cofnąć.'),
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

    final ok = await _appState.deleteCaptureFrame(deviceId, frameIndex);
    if (!mounted) return;
    if (ok) {
      setState(() {
        _framesByDevice[deviceId]?.removeWhere((f) => f.index == frameIndex);
        _cache.remove('$deviceId/$frameIndex');
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
    final hasAny =
        _framesByDevice.values.any((frames) => frames.isNotEmpty);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Zdjęcia pomiarowe'),
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
          : !hasAny
              ? Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.photo_library_outlined,
                          size: 48, color: cs.onSurfaceVariant),
                      const SizedBox(height: 12),
                      Text('Brak zdjęć pomiarowych',
                          style: tt.bodyMedium
                              ?.copyWith(color: cs.onSurfaceVariant)),
                    ],
                  ),
                )
              : ListView(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  children: cameras.map((dev) {
                    final frames = _framesByDevice[dev.deviceId] ?? [];
                    return _DeviceSection(
                      device: dev,
                      frames: frames,
                      loadImage: _loadImage,
                      isLeader: state.isLeader,
                      onDelete: (idx) => _deleteFrame(dev.deviceId, idx),
                      onTap: (idx) => _showViewer(dev.deviceId, idx),
                      cs: cs,
                      tt: tt,
                    );
                  }).toList(),
                ),
    );
  }
}

class _DeviceSection extends StatelessWidget {
  final DeviceInfo device;
  final List<FrameInfo> frames;
  final Future<Uint8List?> Function(String deviceId, int frameIndex) loadImage;
  final bool isLeader;
  final void Function(int frameIndex) onDelete;
  final void Function(int frameIndex) onTap;
  final ColorScheme cs;
  final TextTheme tt;

  const _DeviceSection({
    required this.device,
    required this.frames,
    required this.loadImage,
    required this.isLeader,
    required this.onDelete,
    required this.onTap,
    required this.cs,
    required this.tt,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 8),
          child: Text(
            '${device.deviceId}${device.isLeader ? ' (lider)' : ''}  ·  ${frames.length} zdjęć',
            style: tt.labelMedium?.copyWith(
                color: cs.onSurfaceVariant, letterSpacing: 0.3),
          ),
        ),
        if (frames.isEmpty)
          Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: Text('Brak zdjęć',
                style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
          )
        else
          SizedBox(
            height: 140,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              itemCount: frames.length,
              itemBuilder: (context, i) {
                final frame = frames[i];
                return Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: _CaptureCard(
                    frame: frame,
                    deviceId: device.deviceId,
                    loadImage: loadImage,
                    isLeader: isLeader,
                    onDelete: () => onDelete(frame.index),
                    onTap: () => onTap(frame.index),
                    cs: cs,
                    tt: tt,
                  ),
                );
              },
            ),
          ),
        const Divider(height: 1),
        const SizedBox(height: 8),
      ],
    );
  }
}

class _CaptureCard extends StatefulWidget {
  final FrameInfo frame;
  final String deviceId;
  final Future<Uint8List?> Function(String deviceId, int frameIndex) loadImage;
  final bool isLeader;
  final VoidCallback onDelete;
  final VoidCallback onTap;
  final ColorScheme cs;
  final TextTheme tt;

  const _CaptureCard({
    required this.frame,
    required this.deviceId,
    required this.loadImage,
    required this.isLeader,
    required this.onDelete,
    required this.onTap,
    required this.cs,
    required this.tt,
  });

  @override
  State<_CaptureCard> createState() => _CaptureCardState();
}

class _CaptureCardState extends State<_CaptureCard> {
  late final Future<Uint8List?> _future;

  @override
  void initState() {
    super.initState();
    _future = widget.loadImage(widget.deviceId, widget.frame.index);
  }

  @override
  Widget build(BuildContext context) {
    final cs = widget.cs;
    final tt = widget.tt;

    return SizedBox(
      width: 110,
      child: Column(
        children: [
          Expanded(
            child: Stack(
              children: [
                GestureDetector(
                  onTap: widget.onTap,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: FutureBuilder<Uint8List?>(
                      future: _future,
                      builder: (context, snap) {
                        if (snap.connectionState == ConnectionState.waiting) {
                          return Container(
                            color: cs.surfaceContainerHighest,
                            child: const Center(
                                child:
                                    CircularProgressIndicator(strokeWidth: 2)),
                          );
                        }
                        if (snap.hasData && snap.data != null) {
                          return Image.memory(snap.data!,
                              fit: BoxFit.cover,
                              width: double.infinity,
                              height: double.infinity);
                        }
                        return Container(
                          color: cs.surfaceContainerHighest,
                          child: Icon(Icons.broken_image_outlined,
                              size: 28, color: cs.onSurfaceVariant),
                        );
                      },
                    ),
                  ),
                ),
                if (widget.isLeader)
                  Positioned(
                    top: 2,
                    right: 2,
                    child: GestureDetector(
                      onTap: widget.onDelete,
                      child: Container(
                        decoration: BoxDecoration(
                          color: cs.errorContainer.withValues(alpha: 0.88),
                          shape: BoxShape.circle,
                        ),
                        padding: const EdgeInsets.all(4),
                        child: Icon(Icons.close,
                            size: 14, color: cs.onErrorContainer),
                      ),
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 4),
          Text('#${widget.frame.index}',
              style: tt.labelSmall?.copyWith(color: cs.onSurfaceVariant)),
        ],
      ),
    );
  }
}
