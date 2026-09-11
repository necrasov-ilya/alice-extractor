# Alice Extractor

Локальное извлечение реквизитов из русскоязычных документов на Apple Silicon. Проект использует [Yandex AliceAI-T5-35B-A0.6B](https://huggingface.co/yandex/AliceAI-T5-35B-A0.6B) и собственный путь выполнения для MPS.

Первая задача проекта — счета на оплату. AliceAI будет сравниваться с Qwen и правилами на одном фиксированном наборе документов.

## Репозитории

- [AliceAI-T5-35B-A0.6B-MPS-ExpertInt8](https://huggingface.co/necrasov-ilya/AliceAI-T5-35B-A0.6B-MPS-ExpertInt8) — производные веса для MPS. Карточка опубликована, веса появятся после проверки качества.
- [ru-invoice-extraction-benchmark](https://huggingface.co/datasets/necrasov-ilya/ru-invoice-extraction-benchmark) — 50 синтетических счетов с эталонным JSON.

## Что уже работает

- многотокенный проход вычисляет только 8 выбранных экспертов вместо всех 512
- экспертные матрицы хранятся в int8, остальные параметры остаются в BF16
- команда запуска принимает пути к модели и весам
- замер разделяет загрузку, кодирование, первый шаг декодера и генерацию с кэшем
- набор документов проверяется по схеме, арифметике, манифесту и SHA-256

## Установка

Требуются Mac с Apple Silicon, Python 3.12 и достаточно объединённой памяти для модели.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
```

## Подготовка весов

Команда загружает зафиксированную ревизию AliceAI по одному шарду, квантует экспертные матрицы и удаляет временный исходный шард:

```bash
.venv/bin/alice-quantize-checkpoint \
  --revision a0d71f58c40d6affe461797b30b35ff47f52a5f2
```

Итоговые файлы появятся в `local/aliceai-t5/data/int8`. Конфигурация и код модели сохраняются в `local/aliceai-t5/model`. Подробнее: [docs/quantization.md](docs/quantization.md).

## Запуск

```bash
.venv/bin/alice-extractor \
  --model-dir local/aliceai-t5/model \
  --weights-dir local/aliceai-t5/data/int8 \
  "Кто написал роман Война и мир?"
```

Замер производительности:

```bash
.venv/bin/alice-runtime-benchmark \
  --model-dir local/aliceai-t5/model \
  --weights-dir local/aliceai-t5/data/int8
```

Проверка набора счетов:

```bash
.venv/bin/alice-validate-dataset
```

## Предварительный результат на M1 Max

В одном прогретом запуске с 57 входными и 17 выходными токенами полный проход сократился с 10,282 до 2,561 секунды. Кодирование стало быстрее в 5,54 раза, первый проход декодера — в 4,80 раза. Скорость дальнейшего декодирования почти не изменилась: 19,11 против 19,59 токена в секунду.

Это замер исполняемого пути, а не итог качества модели. Исходные данные находятся в [benchmarks/results/m1-max-64gb-runtime.json](benchmarks/results/m1-max-64gb-runtime.json).

## Локальные файлы

```text
local/
  aliceai-t5/
    model/
    scripts/
    data/int8/
  huggingface/
```

Каталог `local` исключён из Git. Веса не попадут в GitHub.

## Документация

- [PLAN.md](PLAN.md)
- [docs/local-models.md](docs/local-models.md)
- [docs/quantization.md](docs/quantization.md)
- [benchmarks/README.md](benchmarks/README.md)
- [schemas/invoice-v1.schema.json](schemas/invoice-v1.schema.json)

## Лицензия

Код распространяется по Apache License 2.0. AliceAI-T5 принадлежит Yandex LLC и распространяется отдельно. Проект не связан с Yandex. Уведомления находятся в [NOTICE](NOTICE).
