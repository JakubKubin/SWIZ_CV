import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/models.dart';
import '../providers/app_state.dart';

class ResultsScreen extends StatefulWidget {
  const ResultsScreen({super.key});

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  @override
  void initState() {
    super.initState();
    // Fetch measurement if not already loaded
    final state = context.read<AppState>();
    if (state.measurement == null) {
      state.fetchMeasurementNow();
    }
  }

  void _showReport(BuildContext context, String report) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Raport pomiarowy'),
        content: SingleChildScrollView(
          child: Text(
            report,
            style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Zamknij'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Wyniki pomiaru'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => state.fetchMeasurementNow(),
          ),
        ],
      ),
      body: state.isLoading
          ? const Center(child: CircularProgressIndicator())
          : state.measurement == null
              ? _EmptyState(state: state)
              : _ResultsBody(
                  result: state.measurement!,
                  theme: theme,
                  onShowReport: () => _showReport(context, state.measurement!.report),
                ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  final AppState state;

  const _EmptyState({required this.state});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.analytics_outlined, size: 64, color: Colors.grey),
            const SizedBox(height: 16),
            const Text(
              'Brak wyników',
              style: TextStyle(fontSize: 20, color: Colors.grey),
            ),
            const SizedBox(height: 8),
            const Text(
              'Uruchom pomiar ze ekranu Przechwycenia,\nalbo odśwież jeśli pipeline już się zakończył.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey),
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: () => state.fetchMeasurementNow(),
              icon: const Icon(Icons.refresh),
              label: const Text('Pobierz wyniki'),
            ),
            if (state.error != null) ...[
              const SizedBox(height: 16),
              Text(
                state.error!,
                style: const TextStyle(color: Colors.red),
                textAlign: TextAlign.center,
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _ResultsBody extends StatelessWidget {
  final MeasurementResult result;
  final ThemeData theme;
  final VoidCallback onShowReport;

  const _ResultsBody({
    required this.result,
    required this.theme,
    required this.onShowReport,
  });

  @override
  Widget build(BuildContext context) {
    final passed = result.validationPassed;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Validation badge
        Card(
          color: passed ? Colors.green.shade50 : Colors.red.shade50,
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Icon(
                  passed ? Icons.check_circle : Icons.cancel,
                  color: passed ? Colors.green : Colors.red,
                  size: 32,
                ),
                const SizedBox(width: 12),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      passed ? 'Pomiar zaliczony' : 'Pomiar niezaliczony',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: passed ? Colors.green.shade800 : Colors.red.shade800,
                      ),
                    ),
                    if (!passed && result.issues.isNotEmpty)
                      Text(
                        'Problemy: ${result.issues.length}',
                        style: TextStyle(color: Colors.red.shade600),
                      ),
                  ],
                ),
              ],
            ),
          ),
        ),

        const SizedBox(height: 12),

        // Dimension cards
        Text('Wymiary obiektu', style: theme.textTheme.titleMedium),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(child: _DimCard(label: 'Szerokość', valueMm: result.widthMm, color: Colors.blue)),
            const SizedBox(width: 8),
            Expanded(child: _DimCard(label: 'Długość', valueMm: result.lengthMm, color: Colors.teal)),
            const SizedBox(width: 8),
            Expanded(child: _DimCard(label: 'Wysokość', valueMm: result.heightMm, color: Colors.orange)),
          ],
        ),

        const SizedBox(height: 16),

        // Quality metrics
        Text('Jakość pomiaru', style: theme.textTheme.titleMedium),
        const SizedBox(height: 8),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                _MetricRow(
                  label: 'RMS płaszczyzny palety',
                  value: '${result.palletRmsMm.toStringAsFixed(2)} mm',
                  ok: result.palletRmsMm < 30,
                ),
                const Divider(height: 20),
                _MetricRow(
                  label: 'Punkty obiektu',
                  value: result.nObjectPts.toString(),
                  ok: result.nObjectPts > 50,
                ),
                const Divider(height: 20),
                _MetricRow(
                  label: 'Inliery palety',
                  value: result.nPalletInliers.toString(),
                  ok: result.nPalletInliers > 100,
                ),
              ],
            ),
          ),
        ),

        // Issues list
        if (result.issues.isNotEmpty) ...[
          const SizedBox(height: 12),
          Text('Problemy', style: theme.textTheme.titleMedium),
          const SizedBox(height: 8),
          Card(
            color: Colors.red.shade50,
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: result.issues
                    .map(
                      (issue) => Padding(
                        padding: const EdgeInsets.symmetric(vertical: 2),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Icon(Icons.warning_amber, size: 16, color: Colors.red),
                            const SizedBox(width: 6),
                            Expanded(child: Text(issue)),
                          ],
                        ),
                      ),
                    )
                    .toList(),
              ),
            ),
          ),
        ],

        const SizedBox(height: 16),

        // Show report button
        OutlinedButton.icon(
          onPressed: onShowReport,
          icon: const Icon(Icons.description),
          label: const Text('Pokaż pełny raport tekstowy'),
        ),
      ],
    );
  }
}

class _DimCard extends StatelessWidget {
  final String label;
  final double valueMm;
  final Color color;

  const _DimCard({required this.label, required this.valueMm, required this.color});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 8),
        child: Column(
          children: [
            Text(label, style: const TextStyle(fontSize: 12, color: Colors.grey)),
            const SizedBox(height: 6),
            Text(
              '${valueMm.toStringAsFixed(0)}',
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            const Text('mm', style: TextStyle(fontSize: 12, color: Colors.grey)),
          ],
        ),
      ),
    );
  }
}

class _MetricRow extends StatelessWidget {
  final String label;
  final String value;
  final bool ok;

  const _MetricRow({required this.label, required this.value, required this.ok});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(
          ok ? Icons.check : Icons.warning_amber,
          size: 18,
          color: ok ? Colors.green : Colors.orange,
        ),
        const SizedBox(width: 8),
        Expanded(child: Text(label)),
        Text(
          value,
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
      ],
    );
  }
}
