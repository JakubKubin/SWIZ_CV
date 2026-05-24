import 'dart:typed_data';

import 'package:flutter/material.dart';

class ImageEntry {
  final String deviceId;
  final int frameIndex;
  final String label;

  const ImageEntry({
    required this.deviceId,
    required this.frameIndex,
    required this.label,
  });
}

class ImageViewerScreen extends StatefulWidget {
  final List<ImageEntry> images;
  final int initialIndex;
  final Future<Uint8List?> Function(String deviceId, int frameIndex) loadImage;

  const ImageViewerScreen({
    super.key,
    required this.images,
    required this.initialIndex,
    required this.loadImage,
  });

  @override
  State<ImageViewerScreen> createState() => _ImageViewerScreenState();
}

class _ImageViewerScreenState extends State<ImageViewerScreen> {
  late final PageController _pageController;
  late int _currentIndex;

  /// Memoized futures so FutureBuilder doesn't restart on every rebuild.
  final Map<int, Future<Uint8List?>> _futures = {};

  @override
  void initState() {
    super.initState();
    _currentIndex = widget.initialIndex;
    _pageController = PageController(initialPage: widget.initialIndex);
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  Future<Uint8List?> _getFuture(int index) {
    return _futures.putIfAbsent(
      index,
      () => widget.loadImage(
        widget.images[index].deviceId,
        widget.images[index].frameIndex,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final label = widget.images[_currentIndex].label;
    final total = widget.images.length;

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: Text(
          '$label  ·  ${_currentIndex + 1}/$total',
          style: const TextStyle(color: Colors.white, fontSize: 14),
        ),
        leading: IconButton(
          icon: const Icon(Icons.close, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: PageView.builder(
        controller: _pageController,
        itemCount: total,
        onPageChanged: (i) => setState(() => _currentIndex = i),
        itemBuilder: (context, i) {
          return FutureBuilder<Uint8List?>(
            future: _getFuture(i),
            builder: (context, snap) {
              if (snap.connectionState == ConnectionState.waiting) {
                return const Center(
                  child: CircularProgressIndicator(color: Colors.white),
                );
              }
              if (snap.hasData && snap.data != null) {
                return InteractiveViewer(
                  minScale: 0.8,
                  maxScale: 6.0,
                  child: Center(child: Image.memory(snap.data!)),
                );
              }
              return const Center(
                child: Icon(
                  Icons.broken_image_outlined,
                  color: Colors.white54,
                  size: 56,
                ),
              );
            },
          );
        },
      ),
    );
  }
}
