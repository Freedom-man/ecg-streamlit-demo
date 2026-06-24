PATHOLOGY_CLASSES = ["MI", "STTC", "CD", "HYP"]


def interpret_profile(probability_by_class, threshold, borderline_margin=0.05):
    probabilities = {
        class_name: float(probability_by_class.get(class_name, 0.0))
        for class_name in ["NORM", *PATHOLOGY_CLASSES]
    }
    detected = [
        class_name
        for class_name, probability in probabilities.items()
        if probability >= threshold
    ]
    pathology_detected = [
        class_name for class_name in PATHOLOGY_CLASSES if class_name in detected
    ]
    borderline = [
        class_name
        for class_name, probability in probabilities.items()
        if abs(probability - threshold) <= borderline_margin
    ]
    borderline_below = [
        class_name
        for class_name in borderline
        if probabilities[class_name] < threshold
    ]

    common = {
        "threshold": float(threshold),
        "borderline_margin": float(borderline_margin),
        "borderline_interval": [
            max(0.0, float(threshold - borderline_margin)),
            min(1.0, float(threshold + borderline_margin)),
        ],
        "detected_classes": detected,
        "pathology_classes": pathology_detected,
        "borderline_classes": borderline,
        "requires_physician_review": True,
    }

    if "NORM" in detected and pathology_detected:
        classes = ", ".join(pathology_detected)
        return {
            **common,
            "code": "norm_pathology_conflict",
            "level": "conflict",
            "title": "Противоречивый диагностический профиль",
            "message": (
                f"NORM и патологические группы {classes} одновременно превысили порог. "
                "Такой результат нельзя интерпретировать как нормальную ЭКГ. Возможны "
                "перекрывающиеся признаки, пороговая неопределенность или ошибка модели; "
                "требуется проверка исходной записи врачом."
            ),
        }

    if len(pathology_detected) > 1:
        classes = ", ".join(pathology_detected)
        return {
            **common,
            "code": "multiple_pathology",
            "level": "pathology",
            "title": "Обнаружено несколько патологических групп",
            "message": (
                f"Порог превысили группы {classes}. В многометочной постановке такие "
                "сочетания допустимы и могут отражать совместно присутствующие ЭКГ-признаки. "
                "Результат требует врачебной оценки с учетом клинических данных."
            ),
        }

    if len(pathology_detected) == 1:
        class_name = pathology_detected[0]
        near_threshold = class_name in borderline
        suffix = (
            " Вероятность расположена в пограничной зоне относительно выбранного порога."
            if near_threshold
            else ""
        )
        return {
            **common,
            "code": "single_pathology",
            "level": "pathology",
            "title": f"Обнаружены признаки группы {class_name}",
            "message": (
                f"Вероятность группы {class_name} превысила порог {threshold:.2f}. "
                "Это предварительная классификация ЭКГ-признаков, а не клинический диагноз."
                + suffix
            ),
        }

    pathology_borderline = [
        class_name for class_name in borderline_below if class_name in PATHOLOGY_CLASSES
    ]
    if "NORM" in detected:
        if pathology_borderline:
            classes = ", ".join(pathology_borderline)
            return {
                **common,
                "code": "norm_with_borderline_pathology",
                "level": "uncertain",
                "title": "Условно нормальный, но пограничный профиль",
                "message": (
                    f"NORM превысил порог, однако группы {classes} находятся близко к нему. "
                    "Результат не следует считать однозначно нормальным без врачебной проверки."
                ),
            }
        return {
            **common,
            "code": "norm_only",
            "level": "normal",
            "title": "Преобладает группа NORM",
            "message": (
                "Ни одна из четырех выбранных патологических групп не превысила порог. "
                "Это не исключает другие изменения ЭКГ и не заменяет врачебное заключение."
            ),
        }

    if borderline:
        classes = ", ".join(borderline)
        return {
            **common,
            "code": "borderline_no_class",
            "level": "uncertain",
            "title": "Пограничный неопределенный результат",
            "message": (
                f"Ни один класс не превысил порог, но группы {classes} расположены в "
                "пограничной зоне. Требуется проверка ЭКГ и выбранного порога."
            ),
        }

    return {
        **common,
        "code": "no_class_detected",
        "level": "uncertain",
        "title": "Классы выше порога не обнаружены",
        "message": (
            "Ни одна диагностическая группа не превысила выбранный порог. Такой результат "
            "является неопределенным и не должен интерпретироваться как подтверждение нормы."
        ),
    }
