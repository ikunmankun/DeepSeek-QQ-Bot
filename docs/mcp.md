# MCP 工具接入

机器人支持通过 MCP（Model Context Protocol）接入外部工具——agent 在对话和
观察轮里可以像用内置工具一样调用它们。能力由**操作员在配置里静态声明**，
模型不能自己添加；每个清单都经过严格校验（端点、效果分类、作用域）。

## 怎么接入

在 `configs/social.yaml`（或 `social.local.yaml`）的 `mcp:` 段添加清单：

```yaml
mcp:
  - url: "http://127.0.0.1:8300/mcp"        # 固定 HTTPS 或本机 HTTP 端点
    enabled: true
    headers:                                 # 可选：认证头（键值对，不含换行）
      Authorization: "Bearer xxx"
    tools:
      - name: query_weather                  # 小写字母/数字/下划线
        effect: read                          # read | write
        description: 查询城市天气
        parameters:
          type: object
          properties:
            city: {type: string}
        scopes: [group]                       # group | qzone_publish | qzone_comment | profile
```

约束（校验不过会直接拒绝启动该清单）：

- 端点必须是固定 HTTPS（或本机 HTTP），URL 里不得内嵌凭据；
- 每个清单最多 8 个工具；工具名必须匹配 `[a-z][a-z0-9_]{0,63}`；
- 必须声明 effect（read/write）和 scopes（作用域白名单）；
- 认证头是简单的键值对，值里不能有换行。

## 安全模型

- 能力清单是操作员owned：模型/聊天请求不能动态加载工具或改清单；
- MCP 工具走和内置工具相同的 agent 循环、守卫与配额；
- write 类工具受发送额度与作用域约束（同 send_group）。

## 参考

- 实现：`app/agent/mcp_tools.py`（McpToolProvider）
- 接线：`app/agent/mcp_tools.py` 由 harness 工具装配消费（social.yaml `mcp:` 段驱动）
