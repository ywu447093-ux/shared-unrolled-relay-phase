"""Post hoc learning-rate check prompted by deep-model validation collapse.

The original matched arms and their selected checkpoints remain unchanged.
"""
import sys
import suite as s
import calibrated
calibrated.register()

MAPPING={'stage6_lr1e4':'stage6','stage8_lr1e4':'stage8'}
_factory=s.model_for

def register():
 if getattr(s,'stability_models_registered',False):return
 def factory(arm):return _factory(MAPPING.get(arm,arm))
 s.model_for=factory;s.ARMS+=list(MAPPING)
 s.LABELS.update({'stage6_lr1e4':'Six stages, lower learning rate','stage8_lr1e4':'Eight stages, lower learning rate'})
 s.stability_models_registered=True

def main():
 register()
 s.dump(s.OUT/'learning_rate_diagnostic_amendment.json',{
  'trigger':'large validation-score declines observed during the original six- and eight-stage training runs',
  'purpose':'distinguish failure under the common optimizer schedule from an intrinsic depth limitation',
  'arms':MAPPING,'peak_learning_rate':1e-4,'epochs':40,'seed':101,
  'controls':'same model initialization, frozen observations, sample order, optimizer family, loss and checkpoint rule',
  'interpretation':'post hoc sensitivity analysis; retained separately from the twenty-arm matched comparison',
  'script_sha256':s.sha(__file__)})
 sys.argv=['suite.py','--epochs','40','--seed','101','--lr','0.0001','--arms',*MAPPING]
 s.main()
if __name__=='__main__':main()
