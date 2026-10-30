# FocusFlow webpage comparison

The saved base and UI-adapter outputs use the same 212 input tokens, frozen prompt, Qwen/Qwen3-0.6B revision c1899de289a04d12100db370d81485cdf75e47ca, greedy decoding, seed 42, and 3072-token cap. The UI run loaded the final UI adapter at training step 464 and verified it alone was active on 112 LoRA layers. Both runs generated complete documents and their raw outputs were preserved without repairs.

Open http://127.0.0.1:8765/webpage-test with Adapter Lab running for side-by-side previews and source links. Browser access and form submission are restricted by the preview's safety policy.

| Observation | Base | UI adapter |
| --- | --- | --- |
| Tokens / CPU generation time | 1291 / 313.9 s | 1737 / 457.7 s |
| Start Planning scrolls | No | No |
| Signup has a form and confirmation | No | No |
| Email has an associated label | Yes | No |
| Schedule preview | Missing | Missing |
| Requested navy/purple palette | Missing | Missing |

Both pages contain navigation, a hero, three features, and a footer. On the emulated 390 px phone, the UI page's layout viewport expanded to 492 px, so it should not be treated as properly fitting the requested width. The browser check compares scroll width to the expanded viewport, which alone does not detect this issue.

The single paired example does not establish general UI quality or a speed advantage, and is not a CGR repair-and-resume experiment. Results and browser observations are in results/webpage_design_v1. A separate versioned experiment is required for a changed prompt or settings.