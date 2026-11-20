"""Secondary formatting-only diagnostics; primary predictions remain unchanged."""

import re

from comparison_baselines import LABELS, PROMPTS, parse_answer


def format_only_label(answer, prompt_name):
    if prompt_name not in PROMPTS:
        raise ValueError('Unknown prompt')
    if not isinstance(answer, str):
        raise ValueError('Raw output must be a string')
    text = answer.strip()
    if prompt_name == 'explanation':
        markers = list(re.finditer(r'\bfinal:', text, flags=re.IGNORECASE))
        if len(markers) != 1:
            return 'invalid'
        text = text[markers[0].end():].strip()
    normalized = text.removesuffix('.').strip().casefold().replace(' ', '_')
    return normalized if normalized in LABELS else 'invalid'


def normalize_predictions(records, prompt_name):
    """Copy predictions and recover only invalid answers with unambiguous label formatting."""
    if prompt_name not in PROMPTS:
        raise ValueError('Unknown prompt')
    normalized = []
    counts = {'total': len(records), 'strict_invalid': 0, 'format_recovered': 0,
              'recovered_correct': 0, 'recovered_incorrect': 0, 'remaining_invalid': 0}
    for record in records:
        answer = record['raw_output']
        if not isinstance(answer, str):
            raise ValueError('Raw output must be a string')
        strict = parse_answer(answer, prompt_name)
        if record['predicted'] != strict:
            raise ValueError('Saved prediction does not match strict parsing of its raw output')
        if record['gold'] not in LABELS:
            raise ValueError('Unknown gold label')
        prediction = strict
        if strict == 'invalid':
            counts['strict_invalid'] += 1
            prediction = format_only_label(answer, prompt_name)
            if prediction != 'invalid':
                counts['format_recovered'] += 1
                counts['recovered_correct' if prediction == record['gold'] else 'recovered_incorrect'] += 1
        counts['remaining_invalid'] += prediction == 'invalid'
        normalized.append(record | {'strict_predicted': strict, 'predicted': prediction})
    return {'records': normalized, 'counts': counts}
