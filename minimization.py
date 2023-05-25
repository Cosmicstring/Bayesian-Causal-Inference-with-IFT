# Constants for minimization routines

def stages(N_steps):
    if N_steps <= 31:
        stage_1 = 5
        stage_2 = 10
        stage_3 = 20
    elif N_steps == 60:
        stage_1 = 5
        stage_2 = 20
        stage_3 = 50
    elif N_steps > 60:
        stage_1 = 5
        stage_2 = 20
        stage_3 = N_steps-10

    print("STAGES:")
    print("stage1 {}, stage2 {}, stage3 {}:".format(stage_1,stage_2,stage_3))
    return stage_1, stage_2, stage_3
