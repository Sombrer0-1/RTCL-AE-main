import unittest
from experiments.optimization.arrival_cl.control import score_update,quality_train,causal_times


class ArrivalCLControlTests(unittest.TestCase):
    def test_serve_first_protects_request_from_long_update(self):
        infer=lambda b:.005
        self.assertAlmostEqual(score_update([.001],0,.060,infer,.033,.8,.1,True),0)
        self.assertAlmostEqual(score_update([.001],0,.060,infer,.033,.8,.1,False),.8)
        self.assertFalse(quality_train([.001],0,.060,infer,.033,.8,.1)[0])

    def test_early_update_improves_all_safe_requests(self):
        infer=lambda b:.005
        times=[.030,.040,.050,.060]
        train=score_update(times,0,.020,infer,.033,.2,.3,True)
        wait=score_update(times,0,.020,infer,.033,.2,.3,False)
        self.assertAlmostEqual(train,2.0)
        self.assertAlmostEqual(wait,1.7)
        self.assertTrue(quality_train(times,0,.020,infer,.033,.2,.3)[0])

    def test_zero_correctness_tie_explores_training(self):
        self.assertTrue(quality_train([.001],0,.060,lambda b:.005,.033,0,0)[0])

    def test_forecast_keeps_pending_and_uses_past_rate(self):
        values=causal_times([.1,.2,.4,.8],[.4,.8],1)
        self.assertEqual(values[:2],[.4,.8])
        self.assertEqual(values[2:],[1.25,1.5,1.75,2.0])

    def test_warmup_restores_model_optimizer_rng_and_modes(self):
        from copy import deepcopy
        import torch
        from experiments.optimization.arrival_cl.warmup import warm_compute
        previous_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            model=torch.nn.Sequential(torch.nn.Conv2d(3,2,1),torch.nn.BatchNorm2d(2),
                torch.nn.Dropout(.2),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())
            optimizer=torch.optim.Adam(model.parameters(),lr=.01)
            loss=torch.nn.functional.cross_entropy(model(torch.randn(4,3,32,32)),torch.zeros(4,dtype=torch.long))
            loss.backward();optimizer.step();model.eval()
            before={k:v.clone() for k,v in model.state_dict().items()}
            state=deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone()
            durations=warm_compute(model,optimizer,torch.device('cpu'),rounds=3)
            self.assertEqual(len(durations),3)
            self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
            after=optimizer.state_dict()
            for key,values in state['state'].items():
                for field,value in values.items():self.assertTrue(torch.equal(value,after['state'][key][field]))
            self.assertEqual(state['param_groups'],after['param_groups'])
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(all(not module.training for module in model.modules()))
        finally:torch.set_num_threads(previous_threads)

    def test_learning_window_reduces_to_latency_with_perfect_accuracy(self):
        from src.schedulers.window_planner import FixedPlant,plan_window
        from experiments.optimization.arrival_cl.learning_window import LearningPlant,plan_learning_window
        data=dict(trace=[dict(arrival=t) for t in [.02,.04,.18,.31,.52,.64,.82,.95]],
                  compute=[.06],prep=[.008],infer_intercept=.005,infer_slope=0)
        baseline=plan_window(FixedPlant(data),0,0,0,4)
        actual=plan_learning_window(LearningPlant(data,.033,1,0),0,0,0,4)
        self.assertEqual(baseline,actual)

    def test_learning_window_obeys_update_completion_deadlines(self):
        from experiments.optimization.arrival_cl.learning_window import LearningPlant,plan_learning_window
        data=dict(trace=[dict(arrival=t) for t in [.01,.02,.03,.05,.4,.6]],
                  compute=[.06],prep=[.008],infer_intercept=.005,infer_slope=0)
        deadlines=[.1,.3]
        result,feasible,_=plan_learning_window(LearningPlant(data,.033,.7,0),0,0,0,2,
                                               completion_deadlines=deadlines)
        self.assertTrue(feasible)
        self.assertTrue(all(t+.06<=d+1e-9 for t,d in zip(result[5],deadlines)))

    def test_learning_window_scores_early_update_quality(self):
        from experiments.optimization.arrival_cl.learning_window import LearningPlant,plan_learning_window
        data=dict(trace=[dict(arrival=t) for t in [.03,.04,.05,.06]],
                  compute=[.02],prep=[0],infer_intercept=.005,infer_slope=0)
        result,feasible,_=plan_learning_window(LearningPlant(data,.033,.2,.3),0,0,0,1)
        self.assertTrue(feasible)
        self.assertAlmostEqual(result[2],2.0)
        self.assertEqual(result[5],(0,))

    def test_reference_deadlines_recover_from_overdue_nominal_targets(self):
        from experiments.optimization.arrival_cl.learning_window import LearningPlant,plan_learning_window,reference_deadlines
        data=dict(trace=[dict(arrival=t) for t in [.25,.26,.4,.42]],
                  compute=[.05],prep=[.005],infer_intercept=.005,infer_slope=0)
        plant=LearningPlant(data,.033,.5,.01)
        limits=reference_deadlines(plant,.24,0,[.1,.2,.3])
        self.assertTrue(all(t>.24 for t in limits))
        result,feasible,_=plan_learning_window(plant,.24,0,0,3,completion_deadlines=limits)
        self.assertTrue(feasible)
        self.assertTrue(all(t+.05<=d+1e-8 for t,d in zip(result[5],limits)))

    def test_probability_is_bounded(self):
        self.assertAlmostEqual(score_update([.1],0,.02,lambda b:.005,.033,.9,.8,True),1.0)


if __name__=='__main__':unittest.main()
