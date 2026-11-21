# pet/anim —— 自研桌宠动画素材

`webm/` 里的 114 段透明动画取自 [dsh-pet](https://github.com/PC2005-cloud/dsh-pet)
v0.4.8（MIT 协议），本地原始副本在 `pet/dsh-pet/extracted/assets/webm/`。
本目录是自研桌宠壳（`pet/shell`）的运行素材，与 dsh-pet 程序本体无关。

- `pools.json`：动画池与权重。`weights.idle/turn/move` 可调（随机动作=100−其余三项）；
  `restSeconds` 是每段动画播完后的停留区间，调大 = 桌宠更安静。
- `webm/` 不进 git（56MB 二进制），重建机器时先跑：
  `robocopy pet\dsh-pet\extracted\assets\webm pet\anim\webm /E`
  （若 dsh-pet 目录也被清掉，需先从 GitHub release 重新解包。）
