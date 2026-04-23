import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/app_state.dart';
import 'session_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _serverCtrl = TextEditingController();
  final _deviceCtrl = TextEditingController();
  final _macCtrl = TextEditingController();
  final _sidCtrl = TextEditingController();
  bool _busy = false;
  bool? _connectionOk;

  @override
  void initState() {
    super.initState();
    final s = context.read<AppState>();
    _serverCtrl.text = s.serverUrl;
    _deviceCtrl.text = s.deviceId;
    _macCtrl.text = s.mac;
  }

  @override
  void dispose() {
    _serverCtrl.dispose();
    _deviceCtrl.dispose();
    _macCtrl.dispose();
    _sidCtrl.dispose();
    super.dispose();
  }

  void _syncUrl(AppState s) => s.serverUrl = _serverCtrl.text.trim();

  Future<void> _createAndJoin(AppState s) async {
    _syncUrl(s);
    setState(() => _busy = true);
    final ok = await s.createAndJoin(
      _deviceCtrl.text.trim(),
      _macCtrl.text.trim(),
      true,
    );
    setState(() => _busy = false);
    if (ok && mounted) {
      Navigator.push(
          context, MaterialPageRoute(builder: (_) => const SessionScreen()));
    }
  }

  Future<void> _joinExisting(AppState s) async {
    final sid = _sidCtrl.text.trim();
    if (sid.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Podaj ID sesji')),
      );
      return;
    }
    _syncUrl(s);
    setState(() => _busy = true);
    final ok = await s.joinExisting(
      sid,
      _deviceCtrl.text.trim(),
      _macCtrl.text.trim(),
      false,
    );
    setState(() => _busy = false);
    if (ok && mounted) {
      Navigator.push(
          context, MaterialPageRoute(builder: (_) => const SessionScreen()));
    }
  }

  Future<void> _syntheticTest(AppState s) async {
    _syncUrl(s);
    setState(() => _busy = true);
    final result = await s.runSyntheticTest();
    setState(() => _busy = false);
    if (!mounted || result == null) return;
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Test syntetyczny'),
        content: Table(
          columnWidths: const {0: IntrinsicColumnWidth(), 1: FlexColumnWidth()},
          children: [
            _tableRow('Szerokość', '${result.widthMm.toStringAsFixed(0)} mm'),
            _tableRow('Długość', '${result.lengthMm.toStringAsFixed(0)} mm'),
            _tableRow('Wysokość', '${result.heightMm.toStringAsFixed(0)} mm'),
            _tableRow('Walidacja',
                result.validationPassed ? 'Zaliczona' : 'Niezaliczona'),
          ],
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Zamknij')),
        ],
      ),
    );
  }

  TableRow _tableRow(String label, String value) => TableRow(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 0),
            child: Text(label,
                style: const TextStyle(color: Colors.black54, fontSize: 13)),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 12),
            child: Text(value,
                style:
                    const TextStyle(fontWeight: FontWeight.w600, fontSize: 13)),
          ),
        ],
      );

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('StereoVision'),
        actions: [
          Builder(builder: (context) {
            final iconColor = _connectionOk == true
                ? Colors.green
                : _connectionOk == false
                    ? cs.error
                    : null;
            return IconButton(
              icon: Icon(Icons.network_check, color: iconColor),
              tooltip: 'Test połączenia',
              onPressed: _busy
                  ? null
                  : () async {
                      final messenger = ScaffoldMessenger.of(context);

                      state.serverUrl = _serverCtrl.text.trim();
                      final ok = await state.testConnection();
                      setState(() => _connectionOk = ok);

                      messenger.showSnackBar(SnackBar(
                        content: Text(ok
                            ? 'Serwer dostępny'
                            : 'Brak połączenia z serwerem'),
                        backgroundColor: ok ? null : cs.error,
                      ));
                    },
            );
          }),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (state.error != null)
              _ErrorBanner(message: state.error!, onClose: state.clearError),
            _Section(
              title: 'Połączenie',
              child: TextField(
                controller: _serverCtrl,
                decoration: const InputDecoration(
                  labelText: 'Adres serwera',
                  hintText: 'http://192.168.1.1:8000',
                  prefixIcon: Icon(Icons.dns_outlined),
                ),
                keyboardType: TextInputType.url,
              ),
            ),
            const SizedBox(height: 12),
            _Section(
              title: 'Urządzenie',
              child: Column(
                children: [
                  TextField(
                    controller: _deviceCtrl,
                    decoration: const InputDecoration(
                      labelText: 'ID urządzenia',
                      prefixIcon: Icon(Icons.smartphone_outlined),
                    ),
                  ),
                  const SizedBox(height: 10),
                  TextField(
                    controller: _macCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Adres MAC',
                      prefixIcon: Icon(Icons.router_outlined),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            _Section(
              title: 'Nowa sesja',
              child: SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: _busy ? null : () => _createAndJoin(state),
                  icon: _busy
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2))
                      : const Icon(Icons.add),
                  label: const Text('Utwórz sesję i dołącz jako lider'),
                ),
              ),
            ),
            const SizedBox(height: 12),
            _Section(
              title: 'Dołącz do sesji',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  TextField(
                    controller: _sidCtrl,
                    decoration: const InputDecoration(
                      labelText: 'ID sesji',
                      prefixIcon: Icon(Icons.link_outlined),
                    ),
                  ),
                  const SizedBox(height: 8),
                  ElevatedButton.icon(
                    onPressed: _busy ? null : () => _joinExisting(state),
                    icon: const Icon(Icons.login_outlined),
                    label: const Text('Dołącz'),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            _Section(
              title: 'Diagnostyka',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Test syntetyczny uruchamia pełny pipeline obliczeniowy '
                    'na sztucznie wygenerowanych danych — bez kamer.',
                    style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant),
                  ),
                  const SizedBox(height: 10),
                  SizedBox(
                    width: double.infinity,
                    child: OutlinedButton.icon(
                      onPressed: _busy ? null : () => _syntheticTest(state),
                      icon: const Icon(Icons.science_outlined),
                      label: const Text('Uruchom test syntetyczny'),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}

class _Section extends StatelessWidget {
  final String title;
  final Widget child;

  const _Section({required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    final tt = Theme.of(context).textTheme;
    final cs = Theme.of(context).colorScheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title,
                style: tt.labelLarge?.copyWith(
                  color: cs.primary,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.5,
                )),
            const SizedBox(height: 12),
            child,
          ],
        ),
      ),
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  final String message;
  final VoidCallback onClose;

  const _ErrorBanner({required this.message, required this.onClose});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: cs.errorContainer,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: cs.error.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          Icon(Icons.error_outline, color: cs.error, size: 18),
          const SizedBox(width: 10),
          Expanded(
              child: Text(message,
                  style: TextStyle(color: cs.onErrorContainer, fontSize: 13))),
          GestureDetector(
            onTap: onClose,
            child: Icon(Icons.close, color: cs.onErrorContainer, size: 18),
          ),
        ],
      ),
    );
  }
}
