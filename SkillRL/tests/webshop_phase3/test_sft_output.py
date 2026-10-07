import pytest
from test_bank_protocol import module


def test_rank_zero_output_validation_is_shared_without_peer_write_race(tmp_path):
    setup=module('sft').initialize_output
    class Dist:
        def __init__(self, rank, shared): self.rank,self.shared=rank,shared
        def get_rank(self): return self.rank
        def broadcast_object_list(self, message, src):
            if self.rank==0:self.shared[:]=message
            else:message[:]=self.shared
    shared=[]; output=tmp_path/'sft'; recipe={'examples':2553}
    setup(output,recipe,None,Dist(0,shared))
    # Rank0 has already created files; peers must not reject the nonempty output.
    setup(output,recipe,None,Dist(1,shared))
    with pytest.raises(ValueError,match='exists'):
        setup(output,recipe,None,Dist(0,shared))
    with pytest.raises(ValueError,match='exists'):
        setup(output,recipe,None,Dist(1,shared))
    setup(output,recipe,'checkpoint-160',Dist(0,shared))
    setup(output,recipe,'checkpoint-160',Dist(1,shared))
