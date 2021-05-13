"""Test if the macro extraction process keeps the same execution values for the entries."""

import unittest

import config
from program_processing.parsed_to_nx import parsed_to_nx
from program_processing.parse_linear_formula import ParsedLinearFormula, Arg, parse_linear_formula
from program_processing.common import ArgType
from macro_extraction.program import Program, OperationNode
import math_qa.math_qa as mathqa



class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # load all the different mathqa data
        self.mathqa_data = []
        partitions = ['train', 'test', 'dev']
        for part in partitions:
            self.mathqa_data += mathqa.load_dataset(part, config.MATHQA_DIR)

    def test_all_linear_formulas_compile(self):
        for i, entry in enumerate(self.mathqa_data):
            entry: mathqa.RawMathQAEntry
            program = Program.from_linear_formula(entry.linear_formula)
            self.assertIsInstance(program, Program, f"got program of type {type(program)}")

    def test_all_linear_formulas_evaluate(self):
        for i, entry in enumerate(self.mathqa_data):
            entry: mathqa.RawMathQAEntry
            program = Program.from_linear_formula(entry.linear_formula)
            inputs = entry.processed_problem.numbers
            try:
                x = program.eval(inputs)
                self.assertIsInstance(x, (int, float), f"output has to be int of float. got {x}, {type(x)}.")
            except Exception as e:
                self.fail("got exception in execution")

    def test_all_linear_formulas_evaluate_correctly(self):
        for i, entry in enumerate(self.mathqa_data):
            entry: mathqa.RawMathQAEntry
            program = Program.from_linear_formula(entry.linear_formula)
            inputs = entry.processed_problem.numbers
            program_eval = program.eval(inputs)
            linear_formula_eval = entry.processed_linear_formula.eval(inputs)
            self.assertEqual(program_eval, linear_formula_eval, f"evaluation from program={program_eval} is not equal "
                                                                f"to linear formula evaluation={linear_formula_eval}")

    def test_eq_same(self):
        lf1 = 'add(n1,n2)|add(n3,n4)'
        p1 = Program.from_linear_formula(lf1)
        p2 = Program.from_linear_formula(lf1)
        self.assertEqual(p1, p2, "Should be equal")
        self.assertEqual(p2, p1, "Should be equal")

    def test_eq_diff_inputs(self):
        lf1 = 'add(n1, n2)|add(n3,n4)'
        lf2 = 'add(n1,n2)|add(n1,n2)'
        p1 = Program.from_linear_formula(lf1)
        p2 = Program.from_linear_formula(lf2)
        self.assertNotEqual(p1, p2, "should not be equal")

    def test_eq_diff_ops(self):
        lf1 = 'add(n1,n2)|add(n3,n4)'
        lf2 = 'add(n1,n2)|multiply(n3,n4)'
        p1 = Program.from_linear_formula(lf1)
        p2 = Program.from_linear_formula(lf2)
        self.assertNotEqual(p1, p2, "should not be equal")

    def test_eq_order_invariance(self):
        lf1 = 'add(n3,n4)|multiply(n1,n2)'
        lf2 = 'multiply(n1,n2)|add(n3,n4)'
        p1 = Program.from_linear_formula(lf1)
        p2 = Program.from_linear_formula(lf2)
        self.assertEqual(p1, p2, "Should be equal")

        lf1 = 'add(n1,n0)|multiply(#0,n1)'
        lf2 = 'add(n0,n1)|multiply(n1,#0)'
        p1 = Program.from_linear_formula(lf1)
        p2 = Program.from_linear_formula(lf2)
        self.assertEqual(p1, p2, "Should be equal")

        lf1 = 'add(n1,n0)|divide(#0,n1)'
        lf2 = 'add(n1,n0)|divide(n1,#0)'
        p1 = Program.from_linear_formula(lf1)
        p2 = Program.from_linear_formula(lf2)
        self.assertNotEqual(p1, p2, "Should not be equal")

    def test_repr(self):
        lf = 'divide(n0,n1)|divide(#0,#0)'
        p = Program.from_linear_formula(lf)
        self.assertEqual(str(p), lf)

    def test_sub_program(self):
        lf = 'add(n1,n0)|divide(#0,n1)'
        p = Program.from_linear_formula(lf)
        node_subset = {OperationNode(1, 'divide')}
        sub_program, orig_to_input_map = p.sub_program(node_subset, True)
        sub_program: Program
        self.assertEqual(sub_program.get_n_operations(), 1)
        self.assertEqual(sub_program.get_n_inputs(), 2)

    def test_refactor_macro(self):
        add_3_numbers_lf = 'add(n0,n1)|add(#0,n2)'
        add_3_numbers_p = Program.from_linear_formula(add_3_numbers_lf)
        self.assertEqual(add_3_numbers_p.eval([1, 2, 3]), 6)
        macro_dict = {'add3': add_3_numbers_p}
        add_5_numbers_lf = 'add(n0,n1)|add(#0,n2)|add(#1,n3)|add(#2,n4)'
        add_5_numbers_p = Program.from_linear_formula(add_5_numbers_lf)
        inputs = [1, 2, 3, 4, 5]
        self.assertEqual(add_5_numbers_p.eval(inputs), 15)
        add_5_numbers_p = add_5_numbers_p.refactor_macro(
            {OperationNode(1, 'add'), OperationNode(2, 'add')},
            add_3_numbers_p,
            'add3'
        )
        # should fail if macro map is not give
        with self.assertRaises(Exception):
            add_5_numbers_p.eval(inputs)

        self.assertEqual(add_5_numbers_p.eval(inputs, macro_dict), 15)

    def test_iter_program_cut(self):
        target_lf = 'add(n0,n1)|divide(n0,#0)|multiply(#1,n2)'
        target_p = Program.from_linear_formula(target_lf)
        should_get_lf = [
            'multiply(n0,n1)',
            target_lf,
            'divide(n0,n1)|multiply(#0,n2)',
            'add(n0,n1)|divide(n0,#0)',
            'add(n0,n1)',
            'divide(n0,n1)'
        ]
        # marks what was seen
        should_get_p = {Program.from_linear_formula(lf): False for lf in should_get_lf}
        for cut in target_p.function_cut_iterator():
            self.assertIn(cut, should_get_p)
            should_get_p[cut] = True
        # check that all were seen
        self.assertTrue(all(should_get_p.values()))

    def test_self_refactoring(self):
        subset = [10, 32, 333, 765, 1153, 7778, 2910]
        inputs = [1, 2, 3, 4, 5]
        for i in subset:
            entry: mathqa.RawMathQAEntry = self.mathqa_data[i]
            program = Program.from_linear_formula(entry.linear_formula)
            value = program.eval(inputs)
            iterator = program.function_cut_iterator(min_size=3, max_size=5, max_inputs=4, return_subsets=True)
            for macro, subset in iterator:
                macro_dict = {'macro': macro}
                refactored_program = program.refactor_macro(subset, macro, 'macro')
                orig_eval = program.eval(inputs)
                self.assertEqual(orig_eval, value)
                ref_eval = refactored_program.eval(inputs, macro_dict)
                self.assertEqual(ref_eval, orig_eval)

    def test_other_macro_extraction(self):
        subset = [10, 32, 333, 765, 1153, 7778, 2910]
        inputs = [1, 2, 3, 4, 5]
        counter = {}

        for i in subset:
            entry: mathqa.RawMathQAEntry = self.mathqa_data[i]
            program = Program.from_linear_formula(entry.linear_formula)
            iterator = program.function_cut_iterator(min_size=3, max_size=5, max_inputs=4, return_subsets=True)
            for macro, node_subset in iterator:
                if macro not in counter:
                    counter[macro] = []
                counter[macro].append((i, node_subset))
        # find the macro with the highest number of occurrences
        max_oc = 0
        max_macro = None
        for k, v in counter.items():
            oc = len(v)
            if oc > max_oc:
                max_oc = oc
                max_macro = k
        # replace them all
        macro_dict = {'macro': max_macro}
        for i, node_subset in counter[max_macro]:
            entry: mathqa.RawMathQAEntry = self.mathqa_data[i]
            program = Program.from_linear_formula(entry.linear_formula)
            value = program.eval(inputs)
            refactored_program = program.refactor_macro(node_subset, max_macro, 'macro')
            ref_val = refactored_program.eval(inputs, macro_dict)
            self.assertEqual(value, ref_val)

    def test_match_subprogram(self):
        lf1 = 'add(n0,n1)|add(#0,n1)|multiply(#1,n1)'
        lf2 = 'add(n0,n1)|add(#0,n1)|multiply(#1,n1)|add(#2,n2)|add(#3,n2)|multiply(#4,n2)'
        p1 = Program.from_linear_formula(lf1)
        p2 = Program.from_linear_formula(lf2)
        m = Program._match_subprogram(p2, p1)
        self.assertEqual(len(m), 2)

    def test_refactor_macro_without_subset(self):
        lf1 = 'add(n0,n1)|add(#0,n1)|multiply(#1,n1)'
        lf2 = 'add(n0,n1)|add(#0,n1)|multiply(#1,n1)|add(#2,n2)|add(#3,n2)|multiply(#4,n2)'
        p1 = Program.from_linear_formula(lf1)
        p2 = Program.from_linear_formula(lf2)
        p3 = p2.refactor_macro_without_subset(p1, 'p1')
        print(p3)
        macro_dict = {'p1': p1}
        inputs = [
            [10, 11, 12],
            [-1, 3, 7],
            [1, -1, 0]
        ]
        for x in inputs:
            refactored_eval = p3.eval(x, macro_dict)
            normal_eval = p2.eval(x, macro_dict)
            self.assertAlmostEqual(refactored_eval, normal_eval)

    def test_parse_linear_formula(self):
        lf = 'add(n0,n0)|multiply(#0,const_1)|divide(#1,#0)|'
        expected = ParsedLinearFormula(
            op_list=['add', 'multiply', 'divide'],
            arg_list_list=[
                [Arg(ArgType.input, 0), Arg(ArgType.input, 0)],
                [Arg(ArgType.op, 0), Arg(ArgType.const, 'const_1')],
                [Arg(ArgType.op, 1), Arg(ArgType.op, 0)]
            ]
        )
        result = parse_linear_formula(lf)
        self.assertEqual(result, expected)

    def test_parsed_linear_formula_to_nx_graph(self):
        lf = 'add(n0,n0)|multiply(#0,const_1)|divide(#1,#0)|'
        # TODO: not a test
        result = parse_linear_formula(lf)
        result = parsed_to_nx(result, 2)
        print(result)

if __name__ == '__main__':
    unittest.main()
