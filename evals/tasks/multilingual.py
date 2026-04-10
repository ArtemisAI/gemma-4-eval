"""Multilingual eval tasks.

Tests Gemma4's ability to handle non-English languages, code-switching,
and cross-lingual reasoning. Targets known UTF-8 edge cases fixed in
llama.cpp PR #21534 (Korean, Japanese).
"""


def get_tasks() -> list[dict]:
    return [
        {
            "name": "japanese_technical_docs",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "以下のPythonコードのドキュメントを日本語で書いてください。関数の目的、引数、戻り値、"
                        "例外処理について詳しく説明してください。\n\n"
                        "```python\n"
                        "async def retry_with_backoff(fn, max_retries=3, base_delay=1.0, max_delay=30.0):\n"
                        "    for attempt in range(max_retries):\n"
                        "        try:\n"
                        "            return await fn()\n"
                        "        except Exception as e:\n"
                        "            if attempt == max_retries - 1:\n"
                        "                raise\n"
                        "            delay = min(base_delay * (2 ** attempt), max_delay)\n"
                        "            delay += random.uniform(0, delay * 0.1)\n"
                        "            await asyncio.sleep(delay)\n"
                        "```\n\n"
                        "Also provide an English summary at the end."
                    ),
                }
            ],
        },
        {
            "name": "korean_code_review",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "다음 코드를 한국어로 리뷰해주세요. 버그, 성능 문제, 보안 취약점을 찾아주세요.\n\n"
                        "```javascript\n"
                        "app.get('/api/users/:id', async (req, res) => {\n"
                        "  const query = `SELECT * FROM users WHERE id = ${req.params.id}`;\n"
                        "  const user = await db.query(query);\n"
                        "  if (user.rows.length > 0) {\n"
                        "    res.json(user.rows[0]);\n"
                        "  } else {\n"
                        "    res.status(404).json({error: '사용자를 찾을 수 없습니다'});\n"
                        "  }\n"
                        "});\n"
                        "```\n\n"
                        "리뷰는 한국어로, 수정된 코드는 영어 주석과 함께 제공해주세요."
                    ),
                }
            ],
        },
        {
            "name": "spanish_system_architecture",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Diseña una arquitectura de microservicios para un sistema de comercio electrónico "
                        "que maneje 10,000 pedidos por minuto. Incluye:\n"
                        "1. Diagrama de componentes (en texto)\n"
                        "2. Estrategia de comunicación entre servicios\n"
                        "3. Manejo de transacciones distribuidas (patrón Saga)\n"
                        "4. Estrategia de escalamiento horizontal\n"
                        "5. Estimación de capacidad con números concretos\n\n"
                        "Responde en español pero usa términos técnicos en inglés donde sea apropiado."
                    ),
                }
            ],
        },
        {
            "name": "mixed_language_reasoning",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I need you to solve this problem, but I want the solution structured as follows:\n"
                        "- Analysis in English\n"
                        "- 数学的な証明は日本語で (Mathematical proof in Japanese)\n"
                        "- 최종 결론은 한국어로 (Final conclusion in Korean)\n"
                        "- Resumen ejecutivo en español (Executive summary in Spanish)\n\n"
                        "Problem: Prove that for any connected graph G with n vertices and m edges, "
                        "if m > n-1, then G contains at least one cycle. Use proof by contradiction."
                    ),
                }
            ],
        },
        {
            "name": "arabic_rtl_code_generation",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "اكتب دالة بايثون تقوم بتحويل الأرقام العربية إلى كلمات عربية.\n"
                        "مثال: 1234 -> \"ألف ومئتان وأربعة وثلاثون\"\n\n"
                        "المتطلبات:\n"
                        "1. دعم الأرقام من 0 إلى 999,999\n"
                        "2. القواعد النحوية الصحيحة للعدد والمعدود\n"
                        "3. اختبارات وحدة تشمل الحالات الحدية\n\n"
                        "Write the code with English variable names and Arabic docstrings."
                    ),
                }
            ],
        },
        {
            "name": "chinese_algorithm_explanation",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请用中文详细解释红黑树的插入操作，包括以下内容：\n"
                        "1. 所有旋转情况的详细分析（左旋、右旋）\n"
                        "2. 插入后的重新着色规则\n"
                        "3. 用Python实现完整的插入操作\n"
                        "4. 时间复杂度和空间复杂度的证明\n"
                        "5. 与AVL树的性能对比\n\n"
                        "代码注释用英文，解释用中文。"
                    ),
                }
            ],
        },
        {
            "name": "french_debugging",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Analysez ce code Python et trouvez tous les bugs. Expliquez chaque problème "
                        "en français et fournissez le code corrigé.\n\n"
                        "```python\n"
                        "class ConnectionPool:\n"
                        "    def __init__(self, max_size=10):\n"
                        "        self.pool = []\n"
                        "        self.max_size = max_size\n"
                        "        self.lock = threading.Lock()\n\n"
                        "    def get_connection(self):\n"
                        "        if len(self.pool) > 0:\n"
                        "            return self.pool.pop()\n"
                        "        return self._create_connection()\n\n"
                        "    def release(self, conn):\n"
                        "        if len(self.pool) < self.max_size:\n"
                        "            self.pool.append(conn)\n"
                        "```"
                    ),
                }
            ],
        },
    ]
