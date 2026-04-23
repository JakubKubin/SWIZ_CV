import 'package:flutter/material.dart';

abstract final class AppColors {
  // Brand
  static const seed = Color(0xFF1A3F6F);

  // Session state indicators - muted, not garish
  static const stateIdle = Color(0xFF78909C);
  static const stateCalibrating = Color(0xFFA07830);
  static const stateReady = Color(0xFF1A5276);
  static const stateProcessing = Color(0xFF4A5990);
  static const stateDone = Color(0xFF1E7D4F);

  // Dimension cards - unified family, readable on white
  static const dimWidth = Color(0xFF1E5799);
  static const dimLength = Color(0xFF1A6B5A);
  static const dimHeight = Color(0xFF8B6420);

  // Semantic
  static const success = Color(0xFF1E7D4F);
  static const warning = Color(0xFFA07830);
}

Color stateColor(String state, ColorScheme cs) {
  switch (state) {
    case 'IDLE':
      return AppColors.stateIdle;
    case 'CALIBRATING':
      return AppColors.stateCalibrating;
    case 'READY':
      return AppColors.stateReady;
    case 'PROCESSING':
      return AppColors.stateProcessing;
    case 'DONE':
      return AppColors.stateDone;
    default:
      return AppColors.stateIdle;
  }
}

String stateLabel(String state) {
  switch (state) {
    case 'IDLE':
      return 'Oczekiwanie';
    case 'CALIBRATING':
      return 'Kalibracja';
    case 'READY':
      return 'Gotowa';
    case 'PROCESSING':
      return 'Przetwarzanie';
    case 'DONE':
      return 'Zakończona';
    default:
      return state;
  }
}

ThemeData buildTheme() {
  final cs = ColorScheme.fromSeed(
    seedColor: AppColors.seed,
    brightness: Brightness.light,
  );

  return ThemeData(
    useMaterial3: true,
    colorScheme: cs,
    appBarTheme: AppBarTheme(
      centerTitle: false,
      elevation: 0,
      scrolledUnderElevation: 1,
      backgroundColor: cs.surface,
      foregroundColor: cs.onSurface,
      titleTextStyle: TextStyle(
        color: cs.onSurface,
        fontSize: 18,
        fontWeight: FontWeight.w600,
        letterSpacing: 0,
      ),
    ),
    cardTheme: CardThemeData(
      elevation: 0,
      color: cs.surfaceContainerLowest,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
        side: BorderSide(color: cs.outlineVariant.withValues(alpha: 0.6)),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: cs.surfaceContainerLow,
      contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: BorderSide(color: cs.outlineVariant),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: BorderSide(color: cs.outlineVariant),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: BorderSide(color: cs.primary, width: 1.5),
      ),
      errorBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: BorderSide(color: cs.error),
      ),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        elevation: 0,
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
    ),
    dividerTheme: DividerThemeData(
      color: cs.outlineVariant.withValues(alpha: 0.7),
      thickness: 0.5,
    ),
    listTileTheme: const ListTileThemeData(
      contentPadding: EdgeInsets.symmetric(horizontal: 16),
      minVerticalPadding: 10,
    ),
    snackBarTheme: SnackBarThemeData(
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      backgroundColor: cs.inverseSurface,
      contentTextStyle: TextStyle(color: cs.onInverseSurface),
    ),
    switchTheme: SwitchThemeData(
      thumbColor: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.selected)) return cs.primary;
        return cs.outline;
      }),
      trackColor: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.selected)) {
          return cs.primary.withValues(alpha: 0.3);
        }
        return cs.surfaceContainerHighest;
      }),
    ),
  );
}
