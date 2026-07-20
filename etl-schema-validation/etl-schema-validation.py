from typing import Any
from dataclasses import dataclass
from pydantic import BaseModel
import numbers


class Schema(BaseModel):
    column: str
    type: str
    nullable: bool
    min: float = float('-inf')
    max: float = float('inf')


def validate_record(
        record: dict[str, Any], schemas_by_col_name: dict[str, Schema]
) -> list[str]:
    '''Validate the record according to the given rules. Returns
    a list of errors from validation'''
    errors: list[str] = []
    for col_name, cur_schema in schemas_by_col_name.items():
        if col_name not in record:
            errors.append(f'{col_name}: missing')
            continue
        if record[col_name] is None:
            if cur_schema.nullable:
                continue
            errors.append(f'{col_name}: null')
            continue
        if cur_schema.type in ['float', 'int']:
            cur_schema_type = numbers.Real
        else:
            cur_schema_type = eval(cur_schema.type)
        if not isinstance(record[col_name], cur_schema_type):
            errors.append(
                f'{col_name}: expected {cur_schema.type}, '
                f'got {type(record[col_name]).__name__}'
            )
        if not isinstance(record[col_name], float | int):
            continue
        if not (cur_schema.min <= record[col_name] <= cur_schema.max):
            errors.append(f'{col_name}: out of range')
    return errors


def validate_records(
        records: list[dict[str, Any]],
        schema: list[dict[str, Any]]
):
    """
    Validate records against a schema definition.
    In practice, I'd use pydantic for this. And it looks like
    we actually do have access to pydantic, which is nice
    """
    schemas = [Schema(**s) for s in schema]
    schemas_by_col_name = {s.column: s for s in schemas}
    errors_each_record = [
        validate_record(r, schemas_by_col_name) for r in records
    ]
    return [
        (i, len(errors_each_record[i]) == 0, errors_each_record[i])
        for i in range(len(records))
    ]