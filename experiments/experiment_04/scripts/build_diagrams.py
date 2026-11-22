"""Write eight source-linked standard Mermaid views; no training side effects."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'docs'/'research_validation'/'experiment_04'/'diagrams'
OUT.mkdir(parents=True,exist_ok=True)
views={
'01_architecture':('Full experimental architecture', '''flowchart TB
  A["generate_split: independent latent groups"] --> B["100 observed features"]
  B --> C["CoNNMLP.fc1: 100 to 32, gated"]
  C --> D["ReLU"]
  D --> E["CoNNMLP.fc2: 32 to 32, gated"]
  E --> F["ReLU and dense output: one logit"]
  F --> G["BCE loss and shared SGD"]
  H["Gates.at: mlp, aligned, scrambled, uniform"] --> C
  H --> E
  I["Fixed coordinates and schedule; no learned geometry"] --> H
''', ['core.generate_split','core.Gates.at','runner.fit_condition','conn_exp.models.CoNNMLP']),
'02_dataset':('Synthetic data generation', '''flowchart TB
  A["stream: independent seed for train, val, test"] --> B["Five independent Gaussian pairs a,b"]
  B --> C["Ten noisy measurements of a, then ten of b per group"]
  B --> D["target_score: sum of abs(a+b) minus abs(a-b)"]
  D --> E["Threshold at zero; independent label flip probability 0.05"]
  C --> F{Task}
  F -->|structured| G["Keep group order"]
  F -->|negative| H["Seeded column permutation, same for all splits"]
  G --> I["Observed features and labels"]
  H --> I
  E --> I
''', ['core.stream','core.target_score','core.generate_split']),
'03_gating':('Gaussian and control calculations', '''flowchart TB
  A["make_task_geometry: fixed unit-interval coordinates"] --> B["pairwise_abs_distance: destination by source"]
  C["schedule_at: smoothstep tau, then exact release"] --> D{Released?}
  B --> E["gaussian_gate: exp of negative squared distance over 2 tau squared"]
  D -->|no| E
  D -->|yes| F["Exact ones for all conditions"]
  E --> G["Aligned: use Gaussian"]
  E --> H["Scrambled: frozen within-row column permutation"]
  E --> I["Uniform: sqrt of layer mean square"]
  G --> J["GatedLinear.forward: W_eff = W times detached gate"]
  H --> J
  I --> J
  F --> J
''', ['core.Gates.at','conn_exp.geometry.make_task_geometry','conn_exp.geometry.pairwise_abs_distance','conn_exp.gates.gaussian_gate','conn_exp.models.GatedLinear.forward','conn_exp.schedules.schedule_at']),
'04_alignment':('Aligned versus scrambled geometry', '''flowchart TB
  A["Structured feature groups share latent measurements and interactions"] --> B["Aligned: high gate values near each hidden location"]
  C["Same Gaussian row values"] --> D["Scrambled: seeded independent column permutation in each row"]
  B --> E["Gates.audit: exact sorted row equality and float64 energy checks"]
  D --> E
  E --> F["Source assignment and topology change; row energy stays fixed"]
  G["Negative task: permute observed columns, keep assigned coordinates"] --> H["Assigned locality loses its generative relationship"]
''', ['core.generate_split','core.Gates.audit','conn_exp.gates.apply_row_permutation']),
'05_training':('Actual training execution flow', '''flowchart TB
  A["train_run: unique directory, frozen config and environment"] --> B["For each seed: audit gates, save initial state and batches"]
  B --> C["generate_split: training and validation only"]
  C --> D["fit_condition: shared raw initialization and SGD"]
  D --> E["Gates.at, forward, BCE, finite checks, backward, update"]
  E --> F{Completed updates}
  F -->|1120| G["Save pre_release.pt with last active gate"]
  G --> H["Next update uses exact ones"]
  F -->|1600| I["Save final.pt, curves, diagnostics and trained status"]
  H --> E
  F -->|other| E
  E -->|exception| J["Record failed or interrupted status and traceback"]
''', ['runner.train_run','runner.fit_condition','runner.save_checkpoint','core.Gates.at']),
'06_lifecycle':('Experiment lifecycle statechart', '''stateDiagram-v2
  [*] --> Created: create_run
  Created --> Training: train_run
  Training --> Trained: all fits complete
  Training --> Failed: exception recorded
  Training --> Interrupted: keyboard interrupt recorded
  Trained --> Evaluating: evaluate_run creates evaluation directory
  Evaluating --> Evaluated: all predefined endpoints saved
  Evaluating --> EvaluationFailed: exception recorded
  Evaluating --> EvaluationInterrupted: keyboard interrupt recorded
  Evaluated --> Analyzed: analyze
  Analyzed --> [*]
  Failed --> [*]
  Interrupted --> [*]
  EvaluationFailed --> [*]
  EvaluationInterrupted --> [*]
''', ['runner.create_run','runner.train_run','runner.evaluate_run','analysis.analyze']),
'07_evaluation':('Held-out evaluation and reporting', '''flowchart TB
  A["All 80 confirmatory fits marked trained"] --> B["evaluate_run: independently generate test split"]
  B --> C["Load fixed pre-release and final checkpoints"]
  C --> D["Pre-release: original gate"]
  C --> E["Immediate release: same weights, exact ones"]
  C --> F["Final: 1600 updates, exact ones"]
  D --> G["metrics: accuracy, binary F1, BCE; save predictions"]
  E --> G
  F --> G
  G --> H["analyze: verify pairing; ten independent seed differences"]
  H --> I["Primary exact one-sided sign-flip; descriptive 95 percent t interval"]
  H --> J["Six secondary two-sided tests with Holm adjustment"]
  I --> K["Evidence-limited report and artifact integrity checks"]
  J --> K
''', ['runner.evaluate_run','core.metrics','analysis.analyze','analysis.paired','analysis.holm']),
'08_history':('Comparison with previous CoNN experiments', '''flowchart TB
  A["Experiment 01 and 1B: synthetic motifs; aligned versus shuffled gates"] --> D["Experiment 04: new latent-interaction task and explicit negative geometry control"]
  B["Experiment 1C: locality and release endpoints"] --> E["Experiment 04: fixed pre-release, immediate and final endpoints"]
  C["Experiment 02: HAR; metadata geometry"] --> F["Experiment 04: ground-truth generative structure is specified"]
  G["Phase 3B: context-factor diagnostics"] --> H["Experiment 04: no context factors or learned geometry"]
  D --> I["Reuse unmodified gate, geometry, schedule, model and initializer"]
  E --> I
  F --> I
  H --> I
''', ['conn_exp.experiment1b','conn_exp.experiment1c','conn_exp.experiment2','core.generate_split','runner.train_run']),
}
records=[]
for name,(title,code,refs) in views.items():
    p=OUT/(name+'.mmd')
    with p.open('x',encoding='utf-8') as f:
        f.write('%% '+title+'\n')
        f.write('%% Sources: experiments/experiment_04/src/conn_exp04/{core,runner,analysis}.py\n')
        f.write('%% Historical sources: conn_experiment_01/src/conn_exp/{gates,geometry,models,schedules}.py\n')
        f.write('%% Verified references: '+', '.join(refs)+'\n'+code)
    records.append({'name':name,'title':title,'references':refs})
(OUT/'catalog.json').write_text(json.dumps(records,indent=2)+'\n')
print('Wrote',len(records),'Mermaid diagrams')
