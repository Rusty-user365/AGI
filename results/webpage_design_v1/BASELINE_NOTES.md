# Normal-model webpage baseline — 2026-10-30

The original chat-capable Qwen3-0.6B generated this FocusFlow landing page with
all 112 attached LoRA layers disabled. UI adapter inference has not run.

- 1,291 generated tokens; EOS completion; 313.924 seconds; 4.112 tokens/second.
- CPU fp32. Generation timing excludes model loading.
- Complete HTML document, with no manual design/content/behavior repairs.
- Exact protocol: `data/validation/webpage_design_v1.json`; exact input token
  IDs, revision, runtime, settings and output in `base_result.json`.

Browser observations (`base_browser_checks.json`):

| Requested feature | Observation |
|---|---|
| Complete standalone page | Produced; no external assets |
| Desktop and phone layout | Rendered at 1440px and 390px without horizontal overflow |
| Brand, hero, features, footer | Present; three feature cards |
| Navy background / purple accents | Missed: grey background and blue buttons |
| Visual study-schedule preview | Missing |
| Navigation to sections | Links use `href="#"`; section navigation is not implemented |
| Start Planning scrolls to signup | Missing: button has no handler or destination |
| Label associated with email input | Present |
| Valid-email signup confirmation | Missing: input/button are outside a form; no JavaScript supplied |
| Explicit keyboard-focus styling | No custom focus rule supplied; browser defaults may still provide focus indication |

Screenshots: `base_desktop.png`, `base_phone.png`. These are browser renders of
the unchanged generated source. The preview wrapper is tooling, not model
output. One design example does not establish general model quality.

When the UI adapter is ready, reuse the same frozen prompt, chat tokenization,
seed, greedy setting and 3,072-token cap. Save its raw output separately and
apply the same browser observations. Keep this first baseline unchanged.
