# Keyword Spotting Demo

The model used was `ai85net-kws20` and the dataloader was `KWS_Project` (found inside `kws20`). The dataloader used some own keywords and as such had to balanced accordingly (see `weights` in the `KWS_Project`).

The parameters used were:
- Optimizer: Adam
- Batch size: 512
- Epochs: 90
- Initial learning rate: 0.001 (then following the schedule in `/policies/schedule_kws20-90ep.yaml`)
- QAT: default

The synthesized network along with the demo can be found in [the ai8x-synthesis repository](https://github.com/danpfister/ai8x-synthesis/tree/ml-on-mcu-project/kws_project).
