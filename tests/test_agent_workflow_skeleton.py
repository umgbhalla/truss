from truss.workflows import TemporalAgentExecutionWorkflow


def test_workflow_skeleton_accessible():
    """Ensure the workflow skeleton can be imported and instantiated."""

    wf = TemporalAgentExecutionWorkflow()
    assert wf.current_status == "initialising" 
