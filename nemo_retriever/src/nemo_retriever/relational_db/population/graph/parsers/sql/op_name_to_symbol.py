def get_symbol(op_name):
    # "Eq", "NotEq", "And", "Or"
    if op_name.lower() == "PGExp".lower():
        return "^", "numeric"
    if op_name.lower() == "Eq".lower():
        return "=", "boolean"
    if op_name.lower() == "NotEq".lower():
        return "<>", "boolean"
    if op_name.lower() == "LtEq".lower():
        return "<=", "boolean"
    if op_name.lower() == "GtEq".lower():
        return ">=", "boolean"
    if op_name.lower() == "And".lower():
        return "and", "boolean"
    if op_name.lower() == "Or".lower():
        return "or", "boolean"
    if op_name.lower() == "Divide".lower():
        return "/", "numeric"
    if op_name.lower() == "Multiply".lower():
        return "*", "numeric"
    if op_name.lower() == "LIKE".lower():
        return "LIKE", "boolean"
    if op_name.lower() == "UNION".lower():
        return "UNION", "boolean"
    if op_name.lower() == "UNION ALL".lower():
        return "UNION ALL", "boolean"
    if op_name.lower() == "Minus".lower():
        return "-", "numeric"
    if op_name.lower() == "Lt".lower():
        return "<", "boolean"
    if op_name.lower() == "Gt".lower():
        return ">", "boolean"
    if op_name.lower() == "StringConcat".lower():
        return "||", "string"
    if op_name.lower() == "Plus".lower():
        return "+", "numeric"
    if op_name.lower() == "isNull".lower():
        return "is null", "boolean"
    if op_name.lower() == "isNotNull".lower():
        return "is not null", "boolean"
    if op_name.lower() == "Modulo".lower():
        return "%", "numeric"
    return op_name, []
