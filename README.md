# Incidnet

MVP de uma plataforma agent-first para registrar, investigar e acompanhar
incidentes de suporte N2/N3. A aplicação conecta-se ao MCP oficial da NPAW,
preserva a telemetria original em arquivos JSON e usa uma análise estruturada
via OpenAI (com fallback determinístico).

## Requisitos

- Python 3.11+
- Credenciais do MCP configuradas localmente no ambiente
- Opcional: credenciais da OpenAI para análise por LLM

## Instalação e execução

### Local (desenvolvimento)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
uvicorn app.main:app --reload
```

Acesse <http://127.0.0.1:8000>.

### Docker (produção/teste)

```bash
cp .env.example .env
docker compose up -d
```

Acesse <http://localhost:8000>. O volume `incidnet-data` persiste os dados em
`workspace/`. Logs: `docker compose logs -f`. Para parar: `docker compose down`. O diretório `workspace/` é criado
automaticamente e ignorado pelo Git porque contém credenciais e telemetria.

O arquivo `.env` local é carregado na inicialização. Em produção, as mesmas
variáveis podem ser injetadas pelo ambiente de execução.

URL, account code e credenciais nunca são aceitos pela interface nem persistidos
em `workspace.json`. A configuração do MCP é única para toda a aplicação (todos
os workspaces compartilham a mesma conexão) e a aplicação lê essas
configurações exclusivamente do `.env` local (ou de variáveis injetadas pelo
ambiente de execução). O `.env` está ignorado pelo Git, e mensagens de erro
têm segredos redigidos.

## Adapters de MCP

Internamente, a conexão com o MCP passa por um adapter (`app/adapters/npaw.py`
para a NPAW) que implementa uma interface comum (`MCPAdapter`): transporte e
autenticação, `required_tools()`, `query_user()` e `classify()`. Isso existe
para facilitar trocar de provider MCP no futuro sem tocar em `investigator`,
`llm` ou `storage` — mas, por design, **não é configurável pela UI**: a
conexão ativa (`npaw`) e seus parâmetros vêm sempre do `.env`
(`NPAW_MCP_URL`, `NPAW_API_KEY`, `NPAW_ACCOUNT_CODE`), carregados uma única vez
para todos os workspaces via `app.adapters.load_env_mcp_config()`.

## Workspaces

O Incidnet organiza incidentes em **workspaces** — um workspace por produto,
stack, squad ou qualquer outro recorte que faça sentido para o seu time. Cada
workspace tem seu próprio `PROMPT_BASE.md` (regras de domínio) e agenda cron,
mas compartilha a mesma conexão MCP e configuração de LLM (globais, via `.env`).

## Persistência

Toda persistência funcional fica abaixo de:

```text
workspace/
└── workspaces/{workspace_id}/
    ├── workspace.json
    ├── PROMPT_BASE.md
    └── incidents/
        ├── index.json
        └── {incident_id}/
            ├── incident.json
            ├── feed.json
            └── users/{user_id}.json
```

As gravações JSON usam arquivo temporário e substituição atômica para evitar
arquivos parcialmente escritos.

## Análises versionadas

Cada rodada de investigação produz uma análise imutável com identificador
`ana_*`, armazenada em `incidents/{incident_id}/analyses/{analysis_id}/`. Ela
contém o resumo da rodada e o snapshot dos diagnósticos por usuário. A timeline
recebe apenas o resumo com link para a análise; a página do incidente exibe as
análises como cards. O estado em `users/` continua sendo a projeção operacional
usada para resolver ou reabrir usuários.

Agentes externos podem ler a análise mais recente e publicar uma análise
derivada via `GET .../analyses/latest` e `POST .../analyses`, usando
`parent_analysis_id` para manter a cadeia de investigação auditável.

## Investigação

Para cada incidente aberto, o engine:

1. Resolve o adapter de MCP configurado globalmente (via `.env`) e abre uma
   sessão via `adapter.session()` (para a NPAW: SSE com os headers `npaw-*`,
   incluindo `npaw-environment: prod`).
2. Descobre as tools do servidor e exige as que o adapter declarar em
   `required_tools()` (para a NPAW: `npaw_query_data`).
3. Chama `adapter.query_user()`, que para a NPAW consulta as sessões do
   usuário com NQL validada em uma janela móvel configurável (padrão 48h).
4. Usa título e descrição do incidente para selecionar somente as rows do device
   alvo (lógica específica do adapter; para a NPAW, prioridade para
   `player_type` e `player_name`).
5. Consulta, no máximo, três sessões suspeitas em detalhe.
6. Preserva queries, período, escopo, rows e resumo calculado no JSON do usuário
   (formato normalizado, igual para qualquer provider).
7. Solicita classificação `GOOD`, `BAD` ou `INCONCLUSIVE` à OpenAI, usando o
   classificador determinístico do adapter (`adapter.classify()`) como base e
   fallback.
8. Publica o consolidado no feed.

Sem `OPENAI_API_KEY`, a classificação usa regras determinísticas sobre
`errors`, `playsWithError`, `startupError` e `bufferRatio`. Falhas do MCP são
publicadas no feed sem derrubar a aplicação.

O modelo é obrigatório quando a integração LLM estiver habilitada e deve ser
definido em `OPENAI_MODEL`. Sem esse valor, a aplicação permanece no fallback
determinístico. A integração usa a Responses API com saída Pydantic estruturada.

## Scheduler

Cada workspace tem uma expressão cron de cinco campos. O padrão é `0 0 * * *`
no timezone `America/Sao_Paulo`. Apenas incidentes `OPEN` são processados. A
tela do incidente também oferece o botão **Rodar investigação agora**.

## Exclusões

A tela do incidente permite remover individualmente um usuário e seu diagnóstico.
Também é possível excluir um incidente completo pelo painel do workspace ou pela
própria timeline. As duas ações exigem confirmação visual e ficam bloqueadas
enquanto uma investigação do incidente estiver em andamento.

## Testes

```bash
pytest
```

Os testes cobrem a estrutura do workspace, travessia de caminhos, preservação de
segredos, parsing CSV com cabeçalhos duplicados, erro HTTP 400/NQL, classificação
determinística e o fluxo principal da interface.
