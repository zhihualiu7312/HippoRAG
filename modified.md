damping 就按照代码的公式，s=(1−d)r​+dMs  

代码没有base ppr M0

修改M1，加入归一化reset_vec /= reset_vec.sum()  ，M3 中的reset gate也需要归一化吗

Inference 没有跟着 gate_mode 走  ，这一部分修改

M2/M3 inference 有两个实现方案

scores = reset_vec.copy()

for _ in range(max_iter):
    gated_scores = scores * gates
    propagated = M @ gated_scores

    scores = (
        (1 - damping) * reset_vec
        + damping * propagated
    )

reset_vec = normalize(seeds * gates)

scores = reset_vec.copy()

for _ in range(max_iter):
    gated_scores = scores * gates
    propagated = M @ gated_scores

    scores = (
        (1 - damping) * reset_vec
        + damping * propagated
    )


none
adaptive_personalization
gated
personalized_gated

M0:
M1:
M2:
M3:
	​

s
t+1
	​

=(1−d)r+dMs
t
	​

s
t+1
	​

=(1−d)r
q
	​

+dMs
t
	​

s
t+1
	​

=(1−d)r+dM(D
q
	​

s
t
	​

)
s
t+1
	​

=(1−d)r
q
	​

+dM(D
q
	​

s
t
	​

)
	​


其中：

r
q
	​

=Normalize(r⊙g
q
	​

)
