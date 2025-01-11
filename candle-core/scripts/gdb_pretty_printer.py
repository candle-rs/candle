import gdb
import gdb_lookup
import gdb_providers
import functools


def extract_enum(val: gdb.Value):
    key = [x for x in val.type.fields() if x.name is not None][0]
    return val[key]


def extract_rc_value(val: gdb.Value, is_atomic: bool) -> gdb.Value:
    inner = {
        key: value
        for key, value in list(
            gdb_providers.StdRcProvider(val, is_atomic=is_atomic).children()
        )
    }["value"]
    return inner


def extract_vec(val: gdb.Value) -> list[gdb.Value]:
    items = list(gdb_providers.StdVecProvider(val).children())
    return [float(x[1]) for x in items]


class TensorPrinter:
    def __init__(self, val: gdb.Value):
        self._orig_val = val

        # strip out the tensor tag to get to underlying arc
        self._val = val[gdb_providers.ZERO_FIELD]

        # strip out top level arc
        inner = extract_rc_value(self._val, True)

        # get to storage object
        storage = extract_rc_value(inner["storage"], True)["data"]["value"]
        # step through enums to get raw vec
        self._storage_raw_vec = extract_enum(
            extract_enum(storage)[gdb_providers.ZERO_FIELD]
        )[gdb_providers.ZERO_FIELD]

        self._data_ptr = gdb_providers.unwrap_unique_or_non_null(
            self._storage_raw_vec["buf"]["ptr"]
        )

        # get to the shape object
        shape = [
            int(x)
            for x in extract_vec(inner["layout"]["shape"][gdb_providers.ZERO_FIELD])
        ]
        self._shape = shape

    def to_string(self) -> str:
        return "Tensor %s" % list(reversed(self._shape))

    @staticmethod
    def mk_desc(dims: list[int], ptr: gdb.Value, indent: int) -> str:
        str_buf = "["
        dim = dims[0]
        dims_size = functools.reduce(lambda prev, x: prev * x, dims[1:], 1)
        indexes = range(dim)

        if len(indexes) > 10:
            # we select the first three, and last three
            indexes = list(indexes[:3]) + ["..."] + list(indexes[-3:])

        for i, idx in enumerate(indexes):
            if type(idx) == str:
                str_buf += ((indent - 1) * " ") + idx + "\n"
                str_buf += (indent + 1) * " "
                continue

            if len(dims) == 1:
                str_buf += str((ptr + idx).dereference())
                if i != len(indexes) - 1:
                    str_buf += ", "
            else:
                str_buf += TensorPrinter.mk_desc(
                    dims[1:], ptr + (dims_size * idx), indent + 1
                )
                if i != len(indexes) - 1:
                    str_buf += ",\n"
                    str_buf += (indent + 1) * " "

        str_buf += "]"
        return str_buf

    def children(self):
        desc = "\n" + self.mk_desc(self._shape, self._data_ptr, 0)
        yield "raw", self._val
        yield "desc", desc

    @staticmethod
    def display_hint():
        return "string"


def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("candle-core")
    pp.add_printer("Tensor", "^candle_core::tensor::Tensor$", TensorPrinter)
    return pp


gdb.printing.register_pretty_printer(
    gdb.current_objfile(), build_pretty_printer(), replace=True
)
