import 'package:flutter/material.dart';

/// Renderuje wspólne bannery błędu i informacji (jeśli ustawione).
/// Zastępuje powtarzany blok `if (error != null) ... if (info != null) ...`
/// na ekranach sesji, kalibracji i przechwycenia.
class AppBanners extends StatelessWidget {
  final String? error;
  final String? info;
  final VoidCallback onClearError;
  final VoidCallback onClearInfo;

  const AppBanners({
    super.key,
    required this.error,
    required this.info,
    required this.onClearError,
    required this.onClearInfo,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (error != null)
          AppBanner(
            color: cs.errorContainer,
            textColor: cs.onErrorContainer,
            icon: Icons.error_outline,
            message: error!,
            onClose: onClearError,
          ),
        if (info != null)
          AppBanner(
            color: cs.secondaryContainer,
            textColor: cs.onSecondaryContainer,
            icon: Icons.info_outline,
            message: info!,
            onClose: onClearInfo,
          ),
      ],
    );
  }
}

class AppBanner extends StatelessWidget {
  final Color color;
  final Color textColor;
  final IconData icon;
  final String message;
  final VoidCallback onClose;

  const AppBanner({
    super.key,
    required this.color,
    required this.textColor,
    required this.icon,
    required this.message,
    required this.onClose,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      color: color,
      margin: const EdgeInsets.only(bottom: 8),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Row(
          children: [
            Icon(icon, color: textColor, size: 20),
            const SizedBox(width: 8),
            Expanded(child: Text(message, style: TextStyle(color: textColor))),
            IconButton(
              icon: Icon(Icons.close, color: textColor, size: 18),
              onPressed: onClose,
              padding: EdgeInsets.zero,
              constraints: const BoxConstraints(),
            ),
          ],
        ),
      ),
    );
  }
}
