# Hermes `model-guidance`

`model-guidance` ist ein nativer Hermes-Agent-Plugin, das offizielle, modellbezogene
Provider-Empfehlungen in kleine, lokale Runtime-Hinweise übersetzt. Es kopiert keine
kompletten Provider-Dokumentationen in den Prompt.

Der Runtime-Pfad ist:

```text
aktuelle Hermes-Modell-ID
  -> Normalisierung
  -> Provider/Familie/Exact-Profile
  -> Inheritance und negative Overrides
  -> Hermes-Overlap-Filter
  -> konservative Task-Erkennung
  -> Größenlimit
  -> pre_llm_call-Kontext
```

Normale LLM-Turns benötigen **keinen Netzwerkzugriff und keinen zusätzlichen
LLM-Aufruf**. Internet wird ausschließlich bei dem expliziten Befehl
`/model-guidance update` verwendet.

## Was enthalten ist

- native Hermes-Pluginstruktur mit `plugin.yaml` und `__init__.py`;
- per-Turn-Modellauflösung über das tatsächliche `model`-Argument von `pre_llm_call`;
- sichere Normalisierung von `provider/model`, `provider:model`, Plain IDs und
  `openrouter/provider/model`;
- exact, alias, prefix/family und sicherer Fallback-Matcher;
- OpenAI-Profile für `gpt-6-astra`, `gpt-5.6`, `gpt-5.5`, `gpt-5.4`,
  `gpt-5.3-codex`, `gpt-5.2`, `gpt-5.1`, `gpt-5` und `gpt-4.1`;
- nicht-injizierende Familienprofile für Anthropic Claude, Gemini/Gemma, Qwen,
  DeepSeek, Kimi/Moonshot, GLM/Zhipu, Meta/Llama, Mistral und xAI/Grok;
- Trennung von Prompt-Regeln und API-/Runtime-Empfehlungen;
- Hermes-Overlap-Registry mit Quellenhinweis auf die aktuelle Hermes-Implementierung;
- getrennte Managed-Profile und lokale User-Overrides;
- deterministischer Resolver-/Compiler-Cache mit Datei-, Config- und Versions-
  Invalidierung;
- konfigurierbare Rule-, Family- und Task-Limits sowie eine lokale Token-Schätzung;
- deterministische Offline-Tests und simuliertes Model Switching;
- fail-open Verhalten: ein kaputtes Profil darf Hermes nicht stoppen.

## Installation für Hermes

### Variante A: in den aktuell verwendeten Hermes-Home installieren

Der aktuelle Hermes-Loader scannt User-Plugins aus `$HERMES_HOME/plugins/` und lädt
sie nur, wenn sie in `plugins.enabled` aktiviert sind. Kopiere den gesamten Ordner
`model-guidance` dorthin:

```text
$HERMES_HOME/plugins/model-guidance/
├── plugin.yaml
├── __init__.py
├── runtime.py
├── model_guidance_core.py
├── model_guidance_sources.py
├── profiles/
└── sources/
```

Unter Windows ist `$HERMES_HOME` profilabhängig, zum Beispiel:

```text
C:\Users\<user>\AppData\Local\hermes\profiles\code
```

Danach in der Konfiguration aktivieren:

```yaml
plugins:
  enabled:
    - model-guidance
```

Ein Profil behält seine eigene Konfiguration. Wenn mehrere Hermes-Profile oder Bots
verwendet werden, muss das Plugin in jedes aktive `$HERMES_HOME/plugins/` kopiert und
in jedem Profil aktiviert werden. Das Installationsskript dieses Repositories erledigt
das für die vorhandenen Profile:

```bash
python install_hermes.py
```

Das Skript überschreibt keine vorhandenen User-Override-Dateien und verändert keine
Secrets. Es legt nur eine Sicherung vorhandener Plugin-Dateien an, wenn am Ziel bereits
eine ältere Installation dieses Plugins existiert.

### Deaktivieren

```yaml
plugins:
  disabled:
    - model-guidance
```

Oder dauerhaft über die Plugin-Einstellung:

```yaml
plugins:
  entries:
    model-guidance:
      settings:
        enabled: false
```

## Runtime-Integration

Das Plugin registriert nur die offiziellen aktuellen Hermes-Schnittstellen:

- `pre_llm_call`: dynamische, request-lokale Guidance; gültige Rückgabe ist
  `{"context": "..."}`. Hermes injiziert diesen Kontext in die aktuelle User-Nachricht,
  nicht in den stabilen System-Prompt. Dadurch bleibt Prompt-Caching intakt.
- `pre_api_request`: **Observer**. Die aktuelle Hermes-Dokumentation ignoriert den
  Rückgabewert. Das Plugin beobachtet begrenzte Runtime-Metadaten nur für Diagnostik und
  behauptet nicht, API-Parameter zu ändern.
- `on_session_start`: ausschließlich Diagnose; es ist nicht die Quelle der aktiven
  Modellwahl.
- Slash-Command `/model-guidance`: Diagnose und manueller Updatepfad.

Die Modell-ID aus jedem `pre_llm_call` ist autoritativ. Es gibt keinen Prozess-globalen
"aktuellen Modell"-Cache, der einen `/model`-Wechsel überleben und falsche Guidance
injizieren könnte.

Der Hook ruft weder ein Modell noch einen Provider auf. Normalbetrieb besteht nur aus
lokalem Registry-Lookup, deterministischer Regelkompilierung und dem vorhandenen
Hermes-LLM-Request. Unveränderte Modell-/Task-/Config-Kombinationen werden aus einem
begrenzten In-Memory-Compiler-Cache bedient. Resolver-Ergebnisse werden ebenfalls
gecached; Änderungen an Profilen, Overlap-Datei, Plugin-Konfiguration oder Compiler-
Version invalidieren den Cache.

## Profile und Matching

Profile liegen als JSON-kompatibles YAML unter `profiles/<provider>/`. JSON ist absichtlich
zulässig, damit der Normalbetrieb ohne PyYAML-Abhängigkeit funktioniert.

Ein Profile enthält unter anderem:

- `provider`, `model_family`, `exact_model_id`;
- `aliases` und `match_prefixes`;
- `inherits`, `remove_rules` und `negative_overrides`;
- `source_urls`, `source_review_date`, `source_hash`, `source_type`;
- `prompt_recommendations`;
- `runtime_recommendations`;
- Konfidenz und Profilrevision.

Die Priorität ist:

1. exact normalized ID;
2. Alias;
3. expliziter Prefix/Pattern;
4. Provider-/Familienfallback;
5. unbekannt ohne injizierte Guidance.

Beispiele:

```text
gpt-5.6
openai/gpt-5.6
openrouter/openai/gpt-5.6
```

werden, wenn die Identifikation eindeutig ist, auf dasselbe OpenAI-Profil abgebildet.
Aggressives Fuzzy Matching gibt es absichtlich nicht.

### Inheritance und negative Overrides

Ein neueres Modell erbt nicht automatisch alte Prompt-Workarounds. Wenn ein Profil
`inherits` nutzt, werden die Eltern zuerst kompiliert; `remove_rules` und
`negative_overrides` entfernen anschließend veraltete Regeln anhand ihrer stabilen IDs.

## Prompt vs. Runtime

Jede Empfehlung wird als eine dieser Klassen geführt:

- `PROMPT_APPLICABLE`: darf als kompakte Guidance injiziert werden;
- `RUNTIME_APPLICABLE`: gehört in die Hermes-/Provider-Request-Konfiguration und wird
  nicht als Fake-Prompt wie „think harder“ ausgegeben;
- `INFORMATIONAL`: Diagnoseinformation ohne automatische Anwendung;
- `UNSUPPORTED_BY_CURRENT_HERMES`: offiziell sinnvoll, aber über die aktuelle Plugin-API
  nicht sicher anwendbar.

Die aktuelle Hermes-API erlaubt diesem Plugin nicht, über `pre_api_request` etwa
`reasoning.effort`, `verbosity`, Compaction oder neue Tools zu setzen. Solche Hinweise
bleiben deshalb sichtbar, aber werden nicht fälschlich als angewandt ausgegeben.

## Hermes-Overlap

`sources/hermes-overlap.yaml` erfasst Guidance, die Hermes bereits selbst liefert. Solche
Regeln werden gezählt, aber nicht erneut injiziert. Beispiele sind generische
Task-Vervollständigung und Teile der Tool-/Ausführungsdisziplin. Die Registry ist eine
reviewte Kompatibilitätsmetadatei, kein zweiter System-Prompt.

## Task-aware Guidance

Die Task-Erkennung ist deterministisch und konservativ. Sie verwendet die aktuelle
User-Nachricht, vorhandene Tool-Nachrichten, Plattforminformationen und eindeutige
Begriffe wie `coding`, `repository`, `browser`, `research`, `writing`, `subagent` oder
`kanban`. Es wird kein zusätzliches LLM für die Klassifikation aufgerufen.

Mögliche Scopes sind `common`, `coding`, `repository-work`, `research`, `computer-use`,
`writing`, `long-running-agent`, `subagent`, `tool-heavy` und `kanban`.

Das Standardlimit beträgt 3600 Zeichen. Nicht injizierte API-/Runtime-Regeln zählen nicht
zum Prompttext. Über die Plugin-Settings sind zusätzlich `max_rules`,
`max_family_rules` und `max_task_rules` begrenzbar. Regeln werden deterministisch nach
Priorität ausgewählt; bei Überlauf fallen niedrig priorisierte Regeln weg. Die
Tokenzahl wird ohne Provider-Tokenizer grob als `ceil(characters / 4)` geschätzt.

## User-Overrides

Managed-Updates werden unter folgendem Profil-Home gespeichert:

```text
$HERMES_HOME/plugin-data/model-guidance/managed-profiles/<provider>/<model>.yaml
```

Eigene Profile liegen getrennt:

```text
$HERMES_HOME/plugin-data/model-guidance/user-overrides/<provider>/<model>.yaml
```

User-Overrides werden nach Managed-Profilen geladen und gewinnen bei identischer
`provider`/`exact_model_id`-Kombination. `/model-guidance update` schreibt niemals in
den User-Override-Pfad. Ein Override ist ein vollständiges, validiertes Profil; als
Ausgangspunkt kann die entsprechende Datei aus `profiles/` kopiert und anschließend
angepasst werden.

## Offizielle Quellen

`sources/providers.yaml` ist das Quellenregister. Aktuell ist OpenAI als offizielle
Integration eingetragen:

- Model Guidance: <https://developers.openai.com/api/docs/guides/latest-model>
- Models: <https://developers.openai.com/api/docs/models>

Die OpenAI-Profile wurden gegen den offiziellen Model-Guidance-Text geprüft. Die
`source_hash`-Werte dokumentieren den geprüften Quellstand. Die nicht-OpenAI-Fallbacks
injizieren absichtlich nichts, solange kein verlässlich belegtes, provider-spezifisches
Prompting-Profil gepflegt ist. So werden keine Provider-Empfehlungen erfunden.

## Update-Prozess

Normalbetrieb ist offline. Der manuelle Befehl:

```text
/model-guidance update
```

macht ausschließlich Folgendes:

1. liest die lokal erlaubte Quellenregistry;
2. ruft nur HTTPS-Quellen von erlaubten offiziellen OpenAI-Hosts ab;
3. begrenzt Zeit, Redirects und Datenmenge;
4. speichert Rohtext und SHA-256-Metadaten plugin-eigen;
5. extrahiert nur compiler-eigene, vorab bekannte Regelanker;
6. validiert das resultierende Profil vor Aktivierung;
7. schreibt nur nach `managed-profiles/`;
8. meldet `ADDED`, `MODIFIED`, `REMOVED` und Fehler.

`/model-guidance update --offline` testet den Pfad ohne Netzwerkzugriff. Remote-Text wird
nie importiert, als Python ausgeführt, als Tool registriert oder als Anweisung an einen
Compiler interpretiert. Bei Fehlern bleiben bestehende lokale Profile aktiv.

## Diagnosebefehle

```text
/model-guidance status
/model-guidance show
/model-guidance stats
/model-guidance test openai/gpt-5.6
/model-guidance test openrouter/openai/gpt-5.4
/model-guidance models
/model-guidance sources
/model-guidance reload
/model-guidance update
/model-guidance update --offline
```

Besonders wichtig:

```text
/model-guidance test <model-id>
```

kontaktiert **niemals** das Modell und **niemals** den Provider. Es testet nur die lokale
Normalisierung, Provider-/Familienauflösung, Profilwahl, Filterung und Kompilierung.

`status` zeigt unter anderem aktives Modell, normalisierte ID, Provider, Familie, Profil,
Quellen, injizierte und Hermes-unterdrückte Regeln, Runtime-/Unsupported-Hinweise und
die Zeichenzahl, geschätzte Prompt-Tokens und den letzten Cache-Status. `stats` zeigt
zusätzlich explizit, dass im Normalbetrieb null zusätzliche LLM- und Netzwerkaufrufe
stattfinden, sowie Resolver-/Compiler-Cache-Hits und -Misses. `show` trennt:

- `ACTIVE PROMPT GUIDANCE`;
- `HERMES-HANDLED GUIDANCE`;
- `RUNTIME SETTINGS`;
- `UNSUPPORTED RECOMMENDATIONS`;
- `INFORMATIONAL RECOMMENDATIONS`;
- `SOURCE PROVENANCE`.

## Weitere Provider hinzufügen

### Neuer Provider

1. Einen tatsächlich existierenden offiziellen Quelllink verifizieren.
2. Den Provider in `sources/providers.yaml` registrieren.
3. Einen Provider-Adapter in `model_guidance_sources.py` ergänzen, wenn die
   Dokumentstruktur nicht OpenAI entspricht.
4. Tests für erlaubte Hosts, Parsing und Offline-Fehler hinzufügen.
5. Profile unter `profiles/<provider>/` anlegen.
6. Nur offiziell belegte Regeln übernehmen; bei Unsicherheit ein leeres Familienprofil
   verwenden.

### Neue Familie oder Exact-ID

Eine neue Familie erhält ein eigenes `*-family.yaml` mit leerer Prompt-Liste, bis echte
Quellenbelege vorhanden sind. Ein exactes Profil darf eigene Regeln, Source-Metadaten und
negative Overrides tragen. Danach:

```text
/model-guidance reload
/model-guidance test provider/model
/model-guidance show
```

## Entwicklung und Tests

Die Tests benötigen nur lokale Python-Abhängigkeiten:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Die Suite testet unter anderem Registry-Loading, fehlerhafte Profile, Aliase,
OpenRouter-IDs, Provider-/Familien-Matching, Inheritance-/Override-Pfade,
Hermes-Overlap, Task-Filter, Size-Limits, User-Overrides, Model Switching,
unknown models, Runtime-Hooks, Commands, Offline-Updates, SSRF-/HTML-Grenzen,
deterministische Rule-Limits, Cache-Hits und einen Netzwerk-Wächter für den
Normalpfad.

Es gibt bewusst keine Tests gegen echte GPT-, Claude-, Gemini-, Qwen-, DeepSeek-, Kimi-,
GLM-, Llama-, Mistral- oder Grok-Endpunkte. Die IDs werden simuliert; die Qualität des
zugrunde liegenden Modells ist nicht Bestandteil dieses Plugins.

## Einschränkungen

- Hermes `pre_api_request` ist aktuell nur Beobachtung; Runtime-Parameter werden nicht
  verändert.
- Der Compiler-Cache ist pro Plugin-Prozess flüchtig. Nach einem Hermes-Neustart werden
  lokale Profile erneut gelesen; es findet dabei weiterhin kein Netzwerkzugriff statt.
- `/model-guidance update` ist zunächst eine sichere, deterministische Change-/Metadata-
  Aktualisierung. Freiform-LLM-Extraktion aus Providerseiten ist bewusst nicht aktiviert.
- Nicht-OpenAI-Profile sind derzeit sichere Nullprofile ohne zusätzliche Prompt-Injektion.
- Neue Model IDs werden erst nach einem lokalen Profilupdate oder einer passenden
  Familienregel mit provider-spezifischer Guidance behandelt.
- User-Plugins sind in Hermes absichtlich opt-in. Ohne Eintrag in `plugins.enabled` lädt
  Hermes das Plugin nicht.

## Für Coding Agents

Arbeite zuerst in `model_guidance_core.py` und schreibe reine Resolver-/Compiler-Tests.
Ändere Hermes Core nicht für provider-spezifische Logik. Halte `pre_llm_call` schnell und
offline, behandle Dokumentation als untrusted data, bewahre User-Overrides und prüfe
immer den echten Hermes-Hookvertrag. Der normale Hook darf keine Netzwerk-, Subagent-
oder Modellaufrufe hinzufügen. Ein neues Profil darf keine alten Workarounds erben,
solange diese Vererbung nicht ausdrücklich belegt und getestet ist. Nach Änderungen
`python -m pytest -q` und `python -m compileall -q .` ausführen.
