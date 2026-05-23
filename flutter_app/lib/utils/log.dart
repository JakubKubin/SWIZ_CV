// log.dart - lekki, leveled logger dla aplikacji.
//
// Używa dart:developer.log, dzięki czemu wpisy mają nazwę komponentu, poziom
// oraz (dla ostrzeżeń/błędów) obiekt wyjątku i stack trace. W DevTools/konsoli
// widać je z tagiem, co ułatwia debugowanie miejsc, w których coś poszło nie tak.
//
// Użycie:
//   static const _log = Log('AppState');
//   _log.warn('Nie udało się odświeżyć sesji', e, st);

import 'dart:developer' as developer;

class Log {
  final String tag;
  const Log(this.tag);

  /// Zdarzenie informacyjne (normalny przebieg).
  void info(String message) => _emit(message, level: 800);

  /// Ostrzeżenie - coś poszło nie tak, ale aplikacja kontynuuje działanie.
  void warn(String message, [Object? error, StackTrace? stackTrace]) =>
      _emit(message, level: 900, error: error, stackTrace: stackTrace);

  /// Błąd - operacja nie powiodła się.
  void error(String message, [Object? error, StackTrace? stackTrace]) =>
      _emit(message, level: 1000, error: error, stackTrace: stackTrace);

  void _emit(
    String message, {
    required int level,
    Object? error,
    StackTrace? stackTrace,
  }) {
    developer.log(
      message,
      name: tag,
      level: level,
      error: error,
      stackTrace: stackTrace,
    );
  }
}
