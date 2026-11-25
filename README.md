# FINDER

Chat de IA em um único arquivo HTML, sem frameworks, dependências ou servidor próprio.

## Especificação técnica

- **Interface:** HTML5 e CSS3 responsivo, com tema escuro e layout adaptado para celular.
- **Lógica:** JavaScript nativo; as requisições usam Fetch API e `ReadableStream`.
- **Provedores:** Google Gemini (`streamGenerateContent`), Groq e OpenRouter (API compatível com chat completions).
- **Imagens:** cole imagens no campo de mensagem ou selecione arquivos PNG, JPEG e WebP (até 4 imagens de 8 MB cada). As imagens ficam no IndexedDB e são enviadas junto com o texto.
- **Histórico:** conversas persistidas em `localStorage` e imagens em IndexedDB, com criação, seleção, renomeação e exclusão.
- **Continuidade:** a seleção de provedor é independente da conversa. O histórico de mensagens é enviado ao provedor selecionado no próximo envio.
- **Backup:** exportação e importação de conversas e imagens em JSON. Chaves de API não são incluídas no backup.
- **Execução:** abra `index.html` em um navegador moderno. Configure a chave de API e o identificador do modelo em **Provedores e modelos**.

## Privacidade e limitações

As chaves e conversas ficam no armazenamento local do navegador neste dispositivo. Qualquer pessoa com acesso ao mesmo perfil do navegador pode acessá-las; não armazene chaves em dispositivos compartilhados. As chaves são transmitidas diretamente ao provedor escolhido. O uso da API está sujeito às permissões, cotas, modelos disponíveis e políticas do provedor. A interface precisa de conexão à internet para conversar com os provedores.

Para analisar imagens, escolha um modelo com suporte a visão no provedor selecionado. Nem todos os modelos da Groq ou OpenRouter aceitam imagens.

Ao abrir o arquivo por `file://`, o navegador ou a política CORS do provedor pode impedir a conexão. Se isso acontecer, sirva a pasta localmente por HTTP (por exemplo, com a extensão Live Server do VS Code) e tente novamente.

## Modelos iniciais sugeridos

Os modelos configurados inicialmente são sugestões e podem mudar de disponibilidade:

- Gemini: `gemini-3.5-flash-lite`
- Groq: `openai/gpt-oss-120b`
- OpenRouter: `openai/gpt-4o-mini`
