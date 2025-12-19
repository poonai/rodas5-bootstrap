from sympy.parsing.maxima import sub_dict
from scipy.special import jn
from pickle import NONE
import jax
import jax.numpy as jnp
import lineax as lx
import sympy
import optimistix as optx
import jax
import sympy2jax
import time
import modal

jax.config.update("jax_enable_x64", True)


def step9():
    alpha = jnp.zeros((8, 8), dtype=jnp.float64)
    alpha_row_sum = jnp.zeros((8), dtype=jnp.float64)
    beta = jnp.zeros((8, 8), dtype=jnp.float64)
    w = jnp.zeros((8, 8), dtype=jnp.float64)
    beta_row_sum_without_ii = jnp.zeros((8), jnp.float64)
    beta_row_sum = jnp.zeros((8), dtype=jnp.float64)
    b = jnp.zeros((8), dtype=jnp.float64)
    b_tilt = jnp.zeros((8), dtype=jnp.float64)

    # page number 9
    def step1(gamma):
        nonlocal beta, alpha
        beta = beta.at[1, 0].set(0)
        alpha = alpha.at[1, 0].set(3.0 * gamma)

    def step2(gamma, alpha3, alpha4, alpha5, alpha52, alpha65, beta_tilt_5):
        nonlocal alpha_row_sum, beta, beta_row_sum_without_ii, alpha
        alpha_row_sum = alpha_row_sum.at[1].set(alpha[1, 0])
        alpha_row_sum = alpha_row_sum.at[2].set(alpha3)
        alpha_row_sum = alpha_row_sum.at[3].set(alpha4)
        alpha_row_sum = alpha_row_sum.at[4].set(alpha5)
        alpha = alpha.at[4, 1].set(alpha52)
        alpha = alpha.at[5, 4].set(alpha65)

        beta_row_sum_without_ii = beta_row_sum_without_ii.at[4].set(beta_tilt_5)

        beta = beta.at[2, 1].set(
            jnp.power(alpha_row_sum[2] / alpha_row_sum[1], 2)
            * ((alpha_row_sum[2] / 3) - gamma)
        )
        beta_row_sum_without_ii = beta_row_sum_without_ii.at[1].set(beta[1][0])

        beta_row_sum_without_ii = beta_row_sum_without_ii.at[2].set(
            (jnp.float64(9) / jnp.float64(2)) * beta[2, 1]
        )

        beta_row_sum_without_ii = beta_row_sum_without_ii.at[5].set(1 - gamma)
        beta_row_sum_without_ii = beta_row_sum_without_ii.at[6].set(1 - gamma)
        beta_row_sum_without_ii = beta_row_sum_without_ii.at[7].set(1 - gamma)

        alpha_row_sum = alpha_row_sum.at[5].set(jnp.float64(1))
        alpha_row_sum = alpha_row_sum.at[6].set(jnp.float64(1))
        alpha_row_sum = alpha_row_sum.at[7].set(jnp.float64(1))

    def step3(gamma):
        nonlocal beta, beta_row_sum_without_ii
        A = lx.MatrixLinearOperator(
            jnp.array(
                [
                    [
                        (jnp.float64(0.5) * jnp.power(alpha_row_sum[1], 2)),
                        (
                            jnp.float64(0.5) * jnp.power(alpha_row_sum[2], 2)
                            - (jnp.float64(2) * gamma * beta_row_sum_without_ii[2])
                        ),
                        -jnp.power(gamma, 2),
                    ],
                    [jnp.power(alpha_row_sum[1], 2), jnp.power(alpha_row_sum[2], 2), 0],
                    [jnp.power(alpha_row_sum[1], 3), jnp.power(alpha_row_sum[2], 3), 0],
                ]
            )
        )

        b = jnp.array(
            [
                0,
                jnp.power(alpha_row_sum[3], 2)
                * ((alpha_row_sum[3] / jnp.float64(3)) - gamma),
                jnp.power(alpha_row_sum[3], 3)
                * ((alpha_row_sum[3] / jnp.float64(4)) - gamma),
            ]
        )

        sol = lx.linear_solve(A, b, solver=lx.AutoLinearSolver(well_posed=False))

        beta = beta.at[3, 1].set(jnp.float64(sol.value[0]))
        beta = beta.at[3, 2].set(jnp.float64(sol.value[1]))
        beta_row_sum_without_ii = beta_row_sum_without_ii.at[3].set(
            jnp.float64(sol.value[2])
        )

    def step4(gamma):
        nonlocal alpha_row_sum, beta, beta_row_sum_without_ii
        A = lx.MatrixLinearOperator(
            jnp.array(
                [
                    [
                        (jnp.float64(0.5) * jnp.power(alpha_row_sum[1], 2)),
                        (
                            jnp.float64(0.5) * jnp.power(alpha_row_sum[2], 2)
                            - (jnp.float64(2) * gamma * beta_row_sum_without_ii[2])
                        ),
                        (
                            (jnp.float64(0.5) * jnp.power(alpha_row_sum[3], 2))
                            - (2 * gamma * beta_row_sum_without_ii[3])
                            - (beta[3][2] * beta_row_sum_without_ii[2])
                        ),
                    ],
                    [
                        jnp.power(alpha_row_sum[1], 2),
                        jnp.power(alpha_row_sum[2], 2),
                        jnp.power(alpha_row_sum[3], 2),
                    ],
                    [
                        jnp.power(alpha_row_sum[1], 3),
                        jnp.power(alpha_row_sum[2], 3),
                        jnp.power(alpha_row_sum[3], 3),
                    ],
                ]
            )
        )

        b = jnp.array(
            [
                jnp.power(gamma, 2) * beta_row_sum_without_ii[4],
                jnp.power(alpha_row_sum[4], 2) * ((alpha_row_sum[4] / 3) - gamma),
                jnp.power(alpha_row_sum[4], 3) * ((alpha_row_sum[4] / 4) - gamma),
            ]
        )

        sol = lx.linear_solve(A, b, solver=lx.AutoLinearSolver(well_posed=False))

        beta = beta.at[4, 1].set(jnp.float64(sol.value[0]))
        beta = beta.at[4, 2].set(jnp.float64(sol.value[1]))
        beta = beta.at[4, 3].set(jnp.float64(sol.value[2]))

    def step5(gamma):
        nonlocal alpha, beta, beta_row_sum_without_ii

        A = lx.MatrixLinearOperator(
            jnp.array(
                [
                    [
                        jnp.power(alpha_row_sum[1], 2),
                        jnp.power(alpha_row_sum[2], 2),
                        jnp.power(alpha_row_sum[3], 2),
                        jnp.power(alpha_row_sum[4], 2),
                    ],
                    [
                        jnp.power(alpha_row_sum[1], 3),
                        jnp.power(alpha_row_sum[2], 3),
                        jnp.power(alpha_row_sum[3], 3),
                        jnp.power(alpha_row_sum[4], 3),
                    ],
                    [
                        beta_row_sum_without_ii[1],
                        beta_row_sum_without_ii[2],
                        beta_row_sum_without_ii[3],
                        beta_row_sum_without_ii[4],
                    ],
                    [
                        0,
                        0,
                        beta[3][2] * beta_row_sum_without_ii[2],
                        (beta[4][2] * beta_row_sum_without_ii[2])
                        + (beta[4][3] * beta_row_sum_without_ii[3]),
                    ],
                ]
            )
        )

        b = jnp.array(
            [
                jnp.float64(1 / 3) - gamma,
                jnp.float64(1 / 4) - gamma,
                jnp.float64(1 / 2) - (2 * gamma) + jnp.power(gamma, 2),
                jnp.float64(1 / 6)
                - (jnp.float64(3 / 2) * gamma)
                + (3 * jnp.power(gamma, 2))
                - jnp.power(gamma, 3),
            ]
        )

        sol = lx.linear_solve(A, b, solver=lx.AutoLinearSolver(well_posed=False))
        beta = beta.at[5, 1].set(jnp.float64(sol.value[0]))
        beta = beta.at[5, 2].set(jnp.float64(sol.value[1]))
        beta = beta.at[5, 3].set(jnp.float64(sol.value[2]))
        beta = beta.at[5, 4].set(jnp.float64(sol.value[3]))

    def step6(gamma):
        nonlocal beta, beta_row_sum_without_ii

        A = lx.MatrixLinearOperator(
            jnp.array(
                [
                    [
                        beta_row_sum_without_ii[1],
                        beta_row_sum_without_ii[2],
                        beta_row_sum_without_ii[3],
                        beta_row_sum_without_ii[4],
                        beta_row_sum_without_ii[5],
                    ],
                    [
                        jnp.power(alpha_row_sum[1], 2),
                        jnp.power(alpha_row_sum[2], 2),
                        jnp.power(alpha_row_sum[3], 2),
                        jnp.power(alpha_row_sum[4], 2),
                        jnp.float64(1),
                    ],
                    [
                        0,
                        0,
                        beta[3][2] * beta_row_sum_without_ii[2],
                        (beta[4][2] * beta_row_sum_without_ii[2])
                        + (beta[4][3] * beta_row_sum_without_ii[3]),
                        jnp.float64(0.5)
                        - (
                            (2 * gamma * beta_row_sum_without_ii[5])
                            - jnp.power(gamma, 2)
                        ),
                    ],
                    [
                        0,
                        0,
                        0,
                        beta[4][3] * beta[3][2] * beta_row_sum_without_ii[2],
                        jnp.float64(1 / 6)
                        - (jnp.float64(3 / 2) * gamma)
                        + (3 * jnp.power(gamma, 2))
                        - jnp.power(gamma, 3),
                    ],
                ],
            )
        )

        b = jnp.array(
            [
                jnp.float64(1 / 2) - (2 * gamma) + jnp.power(gamma, 2),
                jnp.float64(1 / 3) - gamma,
                jnp.float64(1 / 6)
                - (jnp.float64(3 / 2) * gamma)
                + (3 * jnp.power(gamma, 2))
                - jnp.power(gamma, 3),
                jnp.float64(1 / 24)
                - (jnp.float64(2 / 3) * gamma)
                + (3 * jnp.power(gamma, 2))
                - (4 * jnp.power(gamma, 3))
                + jnp.power(gamma, 4),
            ],
            dtype=jnp.float64,
        )

        sol = lx.linear_solve(A, b, solver=lx.AutoLinearSolver(well_posed=False))
        beta = beta.at[6, 1].set(jnp.float64(sol.value[0]))
        beta = beta.at[6, 2].set(jnp.float64(sol.value[1]))
        beta = beta.at[6, 3].set(jnp.float64(sol.value[2]))
        beta = beta.at[6, 4].set(jnp.float64(sol.value[3]))
        beta = beta.at[6, 5].set(jnp.float64(sol.value[4]))

    def step7(gamma):
        nonlocal beta, beta_row_sum_without_ii, alpha_row_sum, alpha
        A = lx.MatrixLinearOperator(
            jnp.array(
                [
                    [
                        beta_row_sum_without_ii[1],
                        beta_row_sum_without_ii[2],
                        beta_row_sum_without_ii[3],
                        beta_row_sum_without_ii[4],
                        beta_row_sum_without_ii[5],
                        0,
                    ],
                    [
                        jnp.power(alpha_row_sum[1], 2),
                        jnp.power(alpha_row_sum[2], 2),
                        jnp.power(alpha_row_sum[3], 2),
                        jnp.power(alpha_row_sum[4], 2),
                        jnp.float64(1),
                        0,
                    ],
                    [
                        0,
                        0,
                        beta[3][2] * beta_row_sum_without_ii[2],
                        (beta[4][2] * beta_row_sum_without_ii[2])
                        + (beta[4][3] * beta_row_sum_without_ii[3]),
                        jnp.float64(1 / 2)
                        - (2 * gamma * beta_row_sum_without_ii[5])
                        - jnp.power(gamma, 2),
                        jnp.float64(1 / 2) - (2 * gamma) + jnp.power(gamma, 2),
                    ],
                    [
                        0,
                        0,
                        0,
                        beta[4][3] * beta[3][2] * beta_row_sum_without_ii[2],
                        jnp.float64(1 / 6)
                        - (jnp.float64(3 / 2) * gamma)
                        + (3 * jnp.power(gamma, 2) - jnp.power(gamma, 3)),
                        jnp.float64(1 / 6)
                        - (jnp.float64(3 / 2) * gamma)
                        + (3 * jnp.power(gamma, 2) - jnp.power(gamma, 3)),
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        beta[5][4]
                        * beta[4][3]
                        * beta[3][2]
                        * beta_row_sum_without_ii[2],
                        jnp.float64(1 / 24)
                        - (jnp.float64(2 / 3) * gamma)
                        + (3 * jnp.power(gamma, 2))
                        - (4 * jnp.power(gamma, 3))
                        + jnp.power(gamma, 4),
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        beta[6][5]
                        * beta[5][4]
                        * beta[4][3]
                        * beta[3][2]
                        * beta_row_sum_without_ii[2],
                    ],
                ]
            )
        )

        b = jnp.array(
            [
                jnp.float64(1 / 2) - (2 * gamma) + jnp.power(gamma, 2),
                jnp.float64(1 / 3) - gamma,
                jnp.float64(1 / 6)
                - (jnp.float64(3 / 2) * gamma)
                + (3 * jnp.power(gamma, 2))
                - jnp.power(gamma, 3),
                jnp.float64(1 / 24)
                - (jnp.float64(2 / 3) * gamma)
                + (3 * jnp.power(gamma, 2))
                - (4 * jnp.power(gamma, 3))
                + jnp.power(gamma, 4),
                jnp.float64(1 / 120)
                - (jnp.float64(5 / 24) * gamma)
                + jnp.float64(5 / 3) * jnp.power(gamma, 2)
                - (5 * jnp.power(gamma, 3))
                + (5 * jnp.power(gamma, 4))
                - jnp.power(gamma, 5),
                jnp.float64(1 / 720)
                - (jnp.float64(1 / 20) * gamma)
                + (jnp.float64(5 / 8) * jnp.power(gamma, 2))
                - (jnp.float64(10 / 3) * jnp.power(gamma, 3))
                + (jnp.float64(15 / 2) * jnp.power(gamma, 4))
                - (6 * jnp.power(gamma, 5))
                + jnp.power(gamma, 6),
            ]
        )

        sol = lx.linear_solve(A, b, solver=lx.AutoLinearSolver(well_posed=False))
        beta = beta.at[7, 1].set(jnp.float64(sol.value[0]))
        beta = beta.at[7, 2].set(jnp.float64(sol.value[1]))
        beta = beta.at[7, 3].set(jnp.float64(sol.value[2]))
        beta = beta.at[7, 4].set(jnp.float64(sol.value[3]))
        beta = beta.at[7, 5].set(jnp.float64(sol.value[4]))
        beta = beta.at[7, 6].set(jnp.float64(sol.value[5]))

    def organize_constant(gamma):
        nonlocal b, b_tilt, alpha, beta, beta_row_sum, w

        for i in range(8):
            beta = beta.at[i, i].set(gamma)

        for i in range(1, 8):
            first_column = beta_row_sum_without_ii[i] - jnp.sum(beta[i, :1:i])
            beta = beta.at[i, 0].set(first_column)

        for i in range(8):
            beta_row_sum = beta_row_sum.at[i].set(jnp.sum(beta[i]))

        # (3.1)
        for i in range(7):
            b = b.at[i].set(beta[7][i])

        b = b.at[7].set(gamma)

        # (3.2)
        for i in range(6):
            b_tilt = b_tilt.at[i].set(beta[6][i])

        b_tilt = b_tilt.at[6].set(gamma)

        # (3.3)
        for i in range(7):
            alpha = alpha.at[7, i].set(beta[6][i])

        for i in range(6):
            alpha = alpha.at[6, i].set(beta[5][i])

        w = jnp.linalg.inv(beta)

    def substitute_symbols(equation):
        """Apply all substitutions to an equation"""
        # substitute constraint relationships
        alpha3, alpha31, alpha32 = sympy.symbols("α3 α31 α32")
        equation = equation.subs(alpha32, alpha3 - alpha31)
        alpha4, alpha41, alpha42, alpha43 = sympy.symbols("α4 α41 α42 α43")
        equation = equation.subs(alpha43, alpha4 - alpha41 - alpha42)
        # alpha5, alpha51, alpha52, alpha53, alpha54 = sympy.symbols("α5 α51 α52 α53 α54")
        # equation = equation.subs(alpha54, alpha5 - alpha51 - alpha52 - alpha53)
        alpha6, alpha61, alpha62, alpha63, alpha64, alpha65 = sympy.symbols(
            "α6 α61 α62 α63 α64 α65"
        )
        equation = equation.subs(alpha64, alpha6 - alpha62 - alpha63 - alpha65)

        # # substitute b values
        # for i in range(8):
        #     equation = equation.subs(sympy.Symbol(f"b{i + 1}"), b[i].item())

        # # substitute b_tilt values
        # for i in range(8):
        #     equation = equation.subs(sympy.Symbol(f"b_tilt{i + 1}"), b_tilt[i].item())

        # # substitute w values
        # for i in range(8):
        #     for j in range(8):
        #         equation = equation.subs(sympy.Symbol(f"w{i + 1}{j + 1}"), w[i][j].item())

        # # substitute β values
        # for i in range(8):
        #     equation = equation.subs(sympy.Symbol(f"β{i + 1}"), beta_row_sum[i].item())

        # # # substitute α values
        # for i in range(8):
        #     equation = equation.subs(sympy.Symbol(f"α{i + 1}"), alpha_row_sum[i].item())

        # # substitute α_7j
        # for i in range(8):
        #     equation = equation.subs(sympy.Symbol(f"α7{i + 1}"), alpha[6][i].item())

        # # substitute α_8j
        # for i in range(8):
        #     equation = equation.subs(sympy.Symbol(f"α8{i + 1}"), alpha[7][i].item())

        # # substitute free parameters
        # equation = equation.subs(sympy.Symbol("α21"), alpha[1][0].item())
        # equation = equation.subs(sympy.Symbol("α52"), alpha[4][1].item())
        # equation = equation.subs(sympy.Symbol("α65"), alpha[5][4].item())
        # # α_ij is a lower triangular matrix. Fill in the rest of α_ij.
        # # This makes the summation process easier later on. Otherwise, we have to juggle indices
        # # to form a equation without the zero terms.
        # for i in range(8):
        #     for j in range(i, 8):
        #         equation = equation.subs(
        #             sympy.Symbol(f"α{i + 1}{j + 1}"), alpha[i][j].item()
        #         )

        return equation

    def step8_sympy_substitution():
        # Create a comprehensive substitution dictionary
        subs_dict = {}

        # substitute b values
        for i in range(8):
            subs_dict[f"b{i + 1}"] = b[i]

        # substitute b_tilt values
        for i in range(8):
            subs_dict[f"b_tilt{i + 1}"] = b_tilt[i]

        # substitute w values
        for i in range(8):
            for j in range(8):
                subs_dict[f"w{i + 1}{j + 1}"] = w[i][j]

        # substitute β values
        for i in range(8):
            subs_dict[f"β{i + 1}"] = beta_row_sum[i]

        # substitute α values
        for i in range(8):
            subs_dict[f"α{i + 1}"] = alpha_row_sum[i]

        # substitute α_7j and α_8j
        for i in range(8):
            subs_dict[f"α7{i + 1}"] = alpha[6][i]
            subs_dict[f"α8{i + 1}"] = alpha[7][i]

        # substitute free parameters and remaining α_ij
        subs_dict["α21"] = alpha[1][0]
        subs_dict["α52"] = alpha[4][1]
        subs_dict["α65"] = alpha[5][4]
        subs_dict["α61"] = jnp.float64(0) 
        for i in range(8):
            for j in range(8):
                if (j+1) >= (i+1):
                    subs_dict[f"α{i + 1}{j + 1}"] = jnp.float64(0)
                    

        for i in range(8):
            for j in range(8):
                subs_dict[f"β{i + 1}{j + 1}"] = beta[i][j]

        return subs_dict

    def extract_coefficients_and_constant(equation, variables):
        """Extract coefficients for given variables and constant term from equation"""
        coefficients = []
        for var in variables:
            coeff = equation.coeff(var)
            coefficients.append(float(coeff) if coeff is not None else 0.0)

        constant_term = -1 * (equation.as_independent(*equation.free_symbols)[0])
        return jnp.array(coefficients), float(constant_term)

    def step8():
        nonlocal b, b_tilt, beta, alpha_row_sum, alpha, beta_row_sum, w

        variables = [
            "α31",
            "α41",
            "α42",
            "α51",
            "α53",
            "α54",
            "α62",
            "α63",
        ]

        subs = step8_sympy_substitution()

        def expand_and_extract_coeff(equation):
            equation = sympy.expand(equation)
            coefficients = []
            for var in variables:
                coeff = equation.coeff(sympy.Symbol(var))
                mod = sympy2jax.SymbolicModule(coeff)
                coefficients.append(mod(**subs))
            return jnp.array(coefficients)

        def extract_constant_term(equation):
            for var in variables:
                equation = equation.subs(sympy.Symbol(var), 0)
            equation = sympy.expand(equation)
            mod = sympy2jax.SymbolicModule(equation)
            return mod(**subs)

        # equation 1 condition 6
        equation1 = 0
        for i in range(8):
            i_sym = 0
            for j in range(8):
                betaj = sympy.Symbol(f"β{j + 1}")
                i_sym += sympy.Symbol(f"α{i + 1}{j + 1}") * betaj
            b_sym = sympy.Symbol(f"b{i + 1}")
            alpha_sym = sympy.Symbol(f"α{i + 1}")
            equation1 += b_sym * alpha_sym * i_sym

        equation1 = substitute_symbols(equation1)
        equation1 = sympy.simplify(equation1)
        equation1 = sympy.expand(equation1)
        equation1_coefficients, equation1_constant_term = (
            expand_and_extract_coeff(equation1),
            extract_constant_term(equation1),
        )
        equation1_constant_term = (-1 * equation1_constant_term) - jnp.float64(1 / 8)

        # equation 2 condition 10
        equation2 = 0
        for i in range(8):
            i_sym = 0
            for j in range(8):
                betaj = sympy.Symbol(f"β{j + 1}")
                i_sym += sympy.Symbol(f"α{i + 1}{j + 1}") * betaj
            b_sym = sympy.Symbol(f"b{i + 1}")
            alpha_sym = sympy.Symbol(f"α{i + 1}")
            equation2 += b_sym * alpha_sym**2 * i_sym
        equation2 = substitute_symbols(equation2)
        equation2 = sympy.simplify(equation2)
        equation2 = sympy.expand(equation2)

        equation2_coefficients, equation2_constant_term = (
            expand_and_extract_coeff(equation2),
            extract_constant_term(equation2),
        )
        equation2_constant_term = (-1 * equation2_constant_term) - jnp.float64(1 / 10)

        # equation 3 condition 12
        equation3 = 0
        for i in range(8):
            b_sym = sympy.Symbol(f"b{i + 1}")
            alpha_sym = sympy.Symbol(f"α{i + 1}")
            i_sym = 0
            for j in range(8):
                alphaj = sympy.Symbol(f"α{j + 1}")
                i_sym += sympy.Symbol(f"α{i + 1}{j + 1}") * alphaj**2
            equation3 += b_sym * alpha_sym * i_sym
        equation3 = substitute_symbols(equation3)
        equation3 = sympy.simplify(equation3)
        equation3 = sympy.expand(equation3)

        equation3_coefficients, equation3_constant_term = (
            expand_and_extract_coeff(equation3),
            extract_constant_term(equation3),
        )
        equation3_constant_term = (-1 * equation3_constant_term) - jnp.float64(1 / 15)

        # equation 4 condition 19
        equation4 = 0
        for i in range(8):
            j_sym = 0
            for j in range(8):
                k_sym = 0
                for k in range(8):
                    alphak = sympy.Symbol(f"α{k + 1}")
                    w_sym = sympy.Symbol(f"w{j + 1}{k + 1}")
                    k_sym += alphak**2 * w_sym
                alphaj = sympy.Symbol(f"α{i + 1}{j + 1}")
                j_sym += alphaj * k_sym
            b_sym = sympy.Symbol(f"b{i + 1}")
            alpha_sym = sympy.Symbol(f"α{i + 1}")
            equation4 += b_sym * alpha_sym * j_sym
        equation4 = substitute_symbols(equation4)
        equation4 = sympy.simplify(equation4)
        equation4 = sympy.expand(equation4)
        equation4_coefficients, equation4_constant_term = (
            expand_and_extract_coeff(equation4),
            extract_constant_term(equation4),
        )
        equation4_constant_term = (-1 * equation4_constant_term) - jnp.float64(1 / 4)

        # equation 5 condition 23
        equation5 = 0
        for i in range(2, 8):
            j_sym = 0
            for j in range(i):
                k_sym = 0
                for k in range(j + 1):
                    alphak = sympy.Symbol(f"α{k + 1}")
                    w_sym = sympy.Symbol(f"w{j + 1}{k + 1}")
                    k_sym += alphak**3 * w_sym
                alphaj = sympy.Symbol(f"α{i + 1}{j + 1}")
                j_sym += alphaj * k_sym
            b_sym = sympy.Symbol(f"b{i + 1}")
            alpha_sym = sympy.Symbol(f"α{i + 1}")
            equation5 += b_sym * alpha_sym * j_sym
        equation5 = substitute_symbols(equation5)
        equation5 = sympy.simplify(equation5)
        equation5 = sympy.expand(equation5)
        equation5_coefficients, equation5_constant_term = (
            expand_and_extract_coeff(equation5),
            extract_constant_term(equation5),
        )
        equation5_constant_term = (-1 * equation5_constant_term) - jnp.float64(1 / 5)

        # equation 6 condition 28
        equation6 = 0
        for i in range(8):
            j_sym = 0
            for j in range(8):
                k_sym = 0
                for k in range(8):
                    l_sym = 0
                    for L in range(8):
                        l_sym += sympy.Symbol(f"w{k + 1}{L + 1}") * (
                            sympy.Symbol(f"α{L + 1}") ** 2
                        )
                    k_sym += (
                        sympy.Symbol(f"α{j + 1}")
                        * sympy.Symbol(f"α{j + 1}{k + 1}")
                        * l_sym
                    )
                j_sym += sympy.Symbol(f"β{i + 1}{j + 1}") * k_sym
            equation6 += sympy.Symbol(f"b{i + 1}") * j_sym
        equation6 = substitute_symbols(equation6)
        equation6 = sympy.simplify(equation6)
        equation6 = sympy.expand(equation6)
        equation6_coefficients, equation6_constant_term = (
            expand_and_extract_coeff(equation6),
            extract_constant_term(equation6),
        )
        equation6_constant_term = (-1 * equation6_constant_term) - jnp.float64(1 / 20)

        # equation 7 condition 6_tilt
        equation7 = 0
        for i in range(8):
            i_sym = 0
            for j in range(8):
                betaj = sympy.Symbol(f"β{j + 1}")
                i_sym += sympy.Symbol(f"α{i + 1}{j + 1}") * betaj
            b_sym = sympy.Symbol(f"b_tilt{i + 1}")
            alpha_sym = sympy.Symbol(f"α{i + 1}")
            equation7 += b_sym * alpha_sym * i_sym

        equation7 = substitute_symbols(equation7)
        equation7 = sympy.simplify(equation7)
        equation7 = sympy.expand(equation7)
        equation7_coefficients, equation7_constant_term = (
            expand_and_extract_coeff(equation7),
            extract_constant_term(equation7),
        )
        equation7_constant_term = (-1 * equation7_constant_term) - jnp.float64(1 / 8)

        # equation 8 condition 19 tilt
        equation8 = 0
        for i in range(8):
            j_sym = 0
            for j in range(8):
                k_sym = 0
                for k in range(8):
                    alphak = sympy.Symbol(f"α{k + 1}")
                    w_sym = sympy.Symbol(f"w{j + 1}{k + 1}")
                    k_sym += alphak**2 * w_sym
                alphaj = sympy.Symbol(f"α{i + 1}{j + 1}")
                j_sym += alphaj * k_sym
            b_sym = sympy.Symbol(f"b_tilt{i + 1}")
            alpha_sym = sympy.Symbol(f"α{i + 1}")
            equation8 += b_sym * alpha_sym * j_sym
        equation8 = substitute_symbols(equation8)
        equation8 = sympy.simplify(equation8)
        equation8 = sympy.expand(equation8)
        equation8_coefficients, equation8_constant_term = (
            expand_and_extract_coeff(equation8),
            extract_constant_term(equation8),
        )
        equation8_constant_term = (-1 * equation8_constant_term) - jnp.float64(1 / 4)

        lhs = jnp.array(
            [
                equation1_coefficients,
                equation2_coefficients,
                equation3_coefficients,
                equation4_coefficients,
                equation5_coefficients,
                equation6_coefficients,
                equation7_coefficients,
                equation8_coefficients,
            ]
        )
        print(lhs)
        rank = jnp.linalg.matrix_rank(lhs)
        assert rank == 8, f"Rank-deficient system: rank={rank}"
        A = lx.MatrixLinearOperator(lhs)
        b = jnp.asarray(
            [
                equation1_constant_term,
                equation2_constant_term,
                equation3_constant_term,
                equation4_constant_term,
                equation5_constant_term,
                equation6_constant_term,
                equation7_constant_term,
                equation8_constant_term,
            ]
        )

        sol = lx.linear_solve(A, b, solver=lx.AutoLinearSolver(well_posed=False))
        alpha = alpha.at[2, 0].set(sol.value[0])
        alpha = alpha.at[3, 0].set(sol.value[1])
        alpha = alpha.at[3, 1].set(sol.value[2])
        alpha = alpha.at[4, 0].set(sol.value[3])
        alpha = alpha.at[4, 2].set(sol.value[4])
        alpha = alpha.at[5, 3].set(sol.value[5])
        alpha = alpha.at[5, 1].set(sol.value[6])
        alpha = alpha.at[5, 2].set(sol.value[7])

        alpha = alpha.at[5, 0].set(
            alpha_row_sum[5] - alpha[5][1] - alpha[5][1] - alpha[5][2] - alpha[5][4]
        )

    def substitute_without_free_parameter():
        nonlocal alpha_row_sum, beta_row_sum, alpha, beta_row_sum_without_ii, w
        subs_dict = {}
        for i in range(8):
            subs_dict[f"b{i + 1}"] = b[i]

        # substitute α except α3,α4,α5
        for i in range(8):
            if i + 1 not in [3, 4, 5]:
                subs_dict[f"α{i + 1}"] = alpha_row_sum[i]

        for i in range(8):
            subs_dict[f"β{i + 1}"] = beta_row_sum[i]

        for i in range(8):
            subs_dict[f"β'{i + 1}"] = beta_row_sum_without_ii[i]
        # substitute α_ij except α52,α65
        for i in range(8):
            for j in range(8):
                if not (i == 4 and j == 1) and not (i == 5 and j == 4):
                    subs_dict[f"α{i + 1}{j + 1}"] = alpha[i][j]

        for i in range(8):
            for j in range(8):
                subs_dict[f"β{i + 1}{j + 1}"] = beta[i][j]

        for i in range(8):
            for j in range(8):
                subs_dict[f"w{i + 1}{j + 1}"] = w[i][j]
        return subs_dict

    def residue(parameters, args):
        start_time = time.time()
        alpha3, alpha4, alpha5, alpha52, alpha65, beta_tilt_5, gamma = parameters
        step1(gamma)
        print("step1 done", time.time() - start_time)
        step2(gamma, alpha3, alpha4, alpha5, alpha52, alpha65, beta_tilt_5)
        print("step2 done", time.time() - start_time)
        step3(gamma)
        print("step3 done", time.time() - start_time)
        step4(gamma)
        print("step4 done", time.time() - start_time)
        step5(gamma)
        print("step5 done", time.time() - start_time)
        step6(gamma)
        print("step6 done", time.time() - start_time)
        step7(gamma)
        print("step7 done", time.time() - start_time)
        organize_constant(gamma)
        print("organize_constant done", time.time() - start_time)
        step8()
        print("step8 done", time.time() - start_time)
        nonlocal alpha_row_sum, beta_row_sum, alpha, beta_row_sum_without_ii, w
        ## add parameters to subs
        alpha_row_sum = alpha_row_sum.at[2].set(alpha3)
        alpha_row_sum = alpha_row_sum.at[3].set(alpha4)
        alpha_row_sum = alpha_row_sum.at[4].set(alpha5)
        alpha = alpha.at[4, 1].set(alpha52)
        alpha = alpha.at[5, 4].set(alpha65)
        beta_row_sum_without_ii = beta_row_sum_without_ii.at[4].set(beta_tilt_5)

        # condition 9

        def condition9_body(i, acc):
            accum = acc
            return accum + b[i] * (alpha_row_sum[i] ** 4)

        condition9 = jax.lax.fori_loop(0, 8, condition9_body, 0.0)
        eq9 = condition9
        r9 = eq9 - jnp.float64(1 / 5)
        print("condition9 done", time.time() - start_time)

        # condition 11
        def condition_11_body(i, acc):
            prev_step = acc

            def inner_j(j, acc):
                i, prev_step = acc

                def inner_k(k, acc):
                    i, prev_step = acc
                    next_step = prev_step + (alpha[i][k] * beta_row_sum[k])
                    return (i, next_step)

                _, k_accum = jax.lax.fori_loop(0, 8, inner_k, (i, 0.0))
                next_step = prev_step + (alpha[i][j] * beta_row_sum[j] * k_accum)
                return (i, next_step)

            _, j_accum = jax.lax.fori_loop(0, 8, inner_j, (i, 0.0))
            next_step = prev_step + (b[i] * j_accum)
            return next_step

        condition11 = jax.lax.fori_loop(0, 8, condition_11_body, 0.0)
        eq11 = condition11
        r11 = eq11 - jnp.float64(1 / 20)
        print("condition11 done", time.time() - start_time)

        def condition_15_body(i, accum):
            prev_step = accum

            def inner_j(j, accum):
                i, prev_step = accum

                def inner_k(k, accum):
                    j, prev_step = accum
                    next_step = prev_step + beta[j][k] * beta_row_sum[k]
                    return (j, next_step)

                _, k_accum = jax.lax.fori_loop(0, 8, inner_k, (j, 0.0))
                next_step = prev_step + (beta[i][j] * alpha_row_sum[j] * k_accum)
                return (i, next_step)

            _, j_accum = jax.lax.fori_loop(0, 8, inner_j, (i, 0.0))
            next_step = prev_step + (b[i] * j_accum)
            return next_step

        condition15_accum = jax.lax.fori_loop(0, 8, condition_15_body, 0.0)
        condition15 = condition15_accum

        r15 = condition15 - jnp.float64(1 / 40)
        print("condition15 done", time.time() - start_time)
        # condition 24

        def condition24_body(i, accum):
            prev_step = accum

            def inner_j(j, accum):
                i, prev_step = accum

                def inner_k(k, accum):
                    j, prev_step = accum

                    def inner_l(L, accum):
                        k, prev_step = accum
                        next_step = prev_step + (alpha[k][L] * beta_row_sum[L])
                        return (k, next_step)

                    _, l_accum = jax.lax.fori_loop(0, 8, inner_l, (k, 0.0))
                    next_step = prev_step + (w[j][k] * alpha_row_sum[k] * l_accum)
                    return (j, next_step)

                _, k_accum = jax.lax.fori_loop(0, 8, inner_k, (j, 0.0))
                next_step = prev_step + (alpha[i][j] * k_accum)
                return (i, next_step)

            _, j_accum = jax.lax.fori_loop(0, 8, inner_j, (i, 0.0))
            next_step = prev_step + (b[i] * alpha_row_sum[i] * j_accum)
            return next_step

        condition24 = jax.lax.fori_loop(0, 8, condition24_body, 0.0)
        r24 = condition24 - jnp.float64(1 / 10)
        print("condition24 done", time.time() - start_time)

        def condition_25_body(i, accum):
            prev_step = accum

            def inner_j(j, accum):
                i, prev_step = accum

                def inner_k(k, accum):
                    j, prev_step = accum

                    def inner_l(L, accum):
                        k, prev_step = accum

                        def inner_m(m, accum):
                            L, prev_step = accum
                            next_step = prev_step + (
                                w[L][m] * jnp.power(alpha_row_sum[m], 2)
                            )
                            return (L, next_step)

                        _, m_accum = jax.lax.fori_loop(0, 8, inner_m, (L, 0.0))
                        next_step = prev_step + (
                            alpha_row_sum[k] * alpha[k][L] * m_accum
                        )
                        return (k, next_step)

                    _, l_accum = jax.lax.fori_loop(0, 8, inner_l, (k, 0.0))
                    next_step = prev_step + (w[j][k] * l_accum)
                    return (j, next_step)

                _, k_accum = jax.lax.fori_loop(0, 8, inner_k, (j, 0.0))
                next_step = prev_step + (alpha[i][j] * k_accum)
                return (i, next_step)

            _, j_accum = jax.lax.fori_loop(0, 8, inner_j, (i, 0.0))
            next_step = prev_step + (b[i] * alpha_row_sum[i] * j_accum)
            return next_step

        condition25_accum = jax.lax.fori_loop(
            0,
            8,
            condition_25_body,
            0.0,
        )
        condition25 = condition25_accum
        r25 = condition25 - jnp.float64(1 / 5)
        print("condition25 done", time.time() - start_time)
        # condition 26

        def condition_26_body(i, accum):
            prev_step = accum

            def inner_j(j, accum):
                i, prev_step = accum

                def inner_k(k, accum):
                    i, prev_step = accum

                    def inner_l(L, accum):
                        k, prev_step = accum
                        next_step = prev_step + (
                            w[k][L] * jnp.power(alpha_row_sum[L], 2)
                        )
                        return (k, next_step)

                    _, l_accum = jax.lax.fori_loop(0, 8, inner_l, (k, 0.0))
                    next_step = prev_step + (alpha[i][k] * l_accum)
                    return (i, next_step)

                _, k_accum = jax.lax.fori_loop(0, 8, inner_k, (i, 0.0))
                next_step = prev_step + (alpha[i][j] * beta_row_sum[j] * k_accum)
                return (i, next_step)

            _, j_accum = jax.lax.fori_loop(0, 8, inner_j, (i, 0.0))
            next_step = prev_step + (b[i] * j_accum)
            return next_step

        condition26_accum = jax.lax.fori_loop(
            0,
            8,
            condition_26_body,
            0.0,
        )
        condition26 = condition26_accum
        r26 = condition26 - jnp.float64(1 / 10)
        print("condition26 done", time.time() - start_time)

        # condition 29
        def condition_29_body(i, accum):
            prev_step = accum

            def inner_j(j, accum):
                i, prev_step = accum

                def inner_k(k, accum):
                    j, prev_step = accum
                    next_step = prev_step + (w[j][k] * jnp.power(alpha_row_sum[k], 2))
                    return (j, next_step)

                _, k_accum = jax.lax.fori_loop(0, 8, inner_k, (j, 0.0))
                next_step = prev_step + jnp.power(alpha[i][j] * k_accum, 2)
                return (i, next_step)

            _, j_accum = jax.lax.fori_loop(0, 8, inner_j, (i, 0.0))
            next_step = prev_step + (b[i] * j_accum)
            return next_step

        condition29 = jax.lax.fori_loop(0, 8, condition_29_body, 0.0)
        r29 = condition29 - jnp.float64(1 / 5)
        print("condition29 done", time.time() - start_time)

        # condition 44
        def condition_44_body(i, accum):
            prev_step = accum

            def inner_j(j, accum):
                i, prev_step = accum

                def inner_k(k, accum):
                    i, j, prev_step = accum

                    def inner_l(L, accum):
                        i, j, k, prev_step = accum

                        def inner_m(m, accum):
                            i, k, L, prev_step = accum

                            def inner_n(n, accum):
                                m, prev_step = accum

                                def inner_r(r, accum):
                                    n, prev_step = accum
                                    next_step = prev_step + (
                                        w[n][r] * jnp.power(alpha_row_sum[r], 2)
                                    )
                                    return (n, next_step)

                                _, r_accum = jax.lax.fori_loop(0, 8, inner_r, (n, 0.0))
                                next_step = prev_step + (w[m][n] * r_accum)
                                return (m, next_step)

                            _, n_accum = jax.lax.fori_loop(0, 8, inner_n, (m, 0.0))
                            next_step = prev_step + (alpha[i][m] * n_accum)
                            return (i, k, L, next_step)

                        _, _, _, m_accum = jax.lax.fori_loop(
                            0, 8, inner_m, (i, k, L, 0.0)
                        )
                        next_step = prev_step + (
                            w[k][L] * jnp.power(alpha_row_sum[L], 2) * m_accum
                        )
                        return (i, j, k, next_step)

                    _, _, _, l_accum = jax.lax.fori_loop(0, 8, inner_l, (i, j, k, 0.0))
                    next_step = prev_step + (w[j][k] * l_accum)
                    return (i, j, next_step)

                _, _, k_accum = jax.lax.fori_loop(0, 8, inner_k, (i, j, 0.0))
                next_step = prev_step + (alpha[i][j] * k_accum)
                return (i, next_step)

            _, j_accum = jax.lax.fori_loop(0, 8, inner_j, (i, 0.0))
            next_step = prev_step + (b[i] * j_accum)
            return next_step

        condition44_accum = jax.lax.fori_loop(0, 8, condition_44_body, 0.0)
        r44 = condition44_accum - jnp.float64(4 / 3)
        print("condition44 done", time.time() - start_time)

        # condition 46
        def condition_46_body(i, accum):
            prev_step = accum

            def inner_j(j, accum):
                i, prev_step = accum

                def inner_k(k, accum):
                    j, prev_step = accum
                    next_step = prev_step + (w[j][k] * alpha_row_sum[k])
                    return (j, next_step)

                _, k_accum = jax.lax.fori_loop(0, 8, inner_k, (j, 0.0))
                next_step = prev_step + (alpha[i][j] * k_accum)
                return (i, next_step)

            _, j_accum = jax.lax.fori_loop(0, 8, inner_j, (i, 0.0))
            next_step = prev_step + (b[i] * j_accum)
            return next_step

        condition46 = jax.lax.fori_loop(0, 8, condition_46_body, 0.0)
        r46 = condition46 - jnp.float64(1 / 2)
        print("condition46 done", time.time() - start_time)

        # condition 45 (JAX version, remove sympy, use subs for constants/parameters)
        def cond45_body(i, cond45_acc):
            accu = cond45_acc
            return accu + b[i] * (beta_row_sum_without_ii[i] - alpha_row_sum[i])

        condition45 = jax.lax.fori_loop(0, 8, cond45_body, 0.0)
        r45 = condition45 - jnp.float64(0)
        print("condition45 done", time.time() - start_time)
        return jnp.array([r9, r11, r15, r24, r25, r26, r29, r44, r45, r46])

    parameters = (
        jnp.float64(0.5),
        jnp.float64(0.7),
        jnp.float64(0.9),
        jnp.float64(0.3),
        jnp.float64(0.2),
        jnp.float64(0.75),
        jnp.float64(0.25),
    )
    r = residue(parameters, 0)
    print(r)

    # solver = optx.LevenbergMarquardt(
    # rtol=1e-8, atol=1e-8, verbose=frozenset({"step", "accepted", "loss", "step_size"})
    # )

    # sol = optx.least_squares(residue, solver, parameters, args=None)
    # print(sol.value)
    print(alpha)
    print(beta)
    print(b)
    print(b_tilt)


condition25 = 0

# condition25 = sympy.simplify(condition25)
# print(condition25)


app = modal.App("example-inference")
image = modal.Image.from_registry(
    "nvidia/cuda:13.1.0-devel-ubuntu22.04", add_python="3.12"
).uv_pip_install(
    "jax[cuda13]>=0.8.1",
    "lineax>=0.0.8",
    "modal>=1.2.6",
    "optimistix>=0.0.11",
    "sympy>=1.14.0",
    "sympy2jax>=0.0.7",
)


@app.function(gpu="A100", image=image, cpu=8.0, memory=32768, timeout=60 * 40)
def tune():
    from scipy.special import jn
    from pickle import NONE
    import jax
    import jax.numpy as jnp
    import lineax as lx
    import sympy
    import optimistix as optx
    import jax
    import sympy2jax
    import time

    jax.config.update("jax_enable_x64", True)
    step9()


@app.local_entrypoint()
def main():
    # run the function locally
    tune.remote()


tune.local()
