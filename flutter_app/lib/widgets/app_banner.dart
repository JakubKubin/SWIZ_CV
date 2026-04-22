import 'package:flutter/material.dart';

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
