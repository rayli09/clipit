examples = [
    {
        "description": "ManimCELogo",
        "code": """
        from manim import *

        class ManimCELogo(Scene):
            def construct(self):
                self.camera.background_color = "#ece6e2"
                logo_green = "#87c2a5"
                logo_blue = "#525893"
                logo_red = "#e07a5f"
                logo_black = "#343434"
                ds_m = MathTex(r"\mathbb{M}", fill_color=logo_black).scale(7)
                ds_m.shift(2.25 * LEFT + 1.5 * UP)
                circle = Circle(color=logo_green, fill_opacity=1).shift(LEFT)
                square = Square(color=logo_blue, fill_opacity=1).shift(UP)
                triangle = Triangle(color=logo_red, fill_opacity=1).shift(RIGHT)
                logo = VGroup(triangle, square, circle, ds_m)  # order matters
                logo.move_to(ORIGIN)
                self.add(logo)
        """,
    },
    {
        "description": "BraceAnnotation",
        "code": """

        from manim import *

        class BraceAnnotation(Scene):
            def construct(self):
                dot = Dot([-2, -1, 0])
                dot2 = Dot([2, 1, 0])
                line = Line(dot.get_center(), dot2.get_center()).set_color(ORANGE)
                b1 = Brace(line)
                b1text = b1.get_text("Horizontal distance")
                b2 = Brace(line, direction=line.copy().rotate(PI / 2).get_unit_vector())
                b2text = b2.get_tex("x-x_1")
                self.add(line, dot, dot2, b1, b2, b1text, b2text)
        """,
    },
    {
        "description": "Vector Arrow",
        "code": """
        from manim import *

        class VectorArrow(Scene):
            def construct(self):
                dot = Dot(ORIGIN)
                arrow = Arrow(ORIGIN, [2, 2, 0], buff=0)
                numberplane = NumberPlane()
                origin_text = Text('(0, 0)').next_to(dot, DOWN)
                tip_text = Text('(2, 2)').next_to(arrow.get_end(), RIGHT)
                self.add(numberplane, dot, arrow, origin_text, tip_text)
        """,
    },
    {
        "description": "GradientImageFromArray",
        "code": """
        from manim import *

        class GradientImageFromArray(Scene):
            def construct(self):
                n = 256
                imageArray = np.uint8(
                    [[i * 256 / n for i in range(0, n)] for _ in range(0, n)]
                )
                image = ImageMobject(imageArray).scale(2)
                image.background_rectangle = SurroundingRectangle(image, color=GREEN)
                self.add(image, image.background_rectangle)
        """,
    },
    {
        "description": "boolean operation",
        "code": """
        class BooleanOperations(Scene):
        def construct(self):
            ellipse1 = Ellipse(
                width=4.0, height=5.0, fill_opacity=0.5, color=BLUE, stroke_width=10
            ).move_to(LEFT)
            ellipse2 = ellipse1.copy().set_color(color=RED).move_to(RIGHT)
            bool_ops_text = MarkupText("<u>Boolean Operation</u>").next_to(ellipse1, UP * 3)
            ellipse_group = Group(bool_ops_text, ellipse1, ellipse2).move_to(LEFT * 3)
            self.play(FadeIn(ellipse_group))

            i = Intersection(ellipse1, ellipse2, color=GREEN, fill_opacity=0.5)
            self.play(i.animate.scale(0.25).move_to(RIGHT * 5 + UP * 2.5))
            intersection_text = Text("Intersection", font_size=23).next_to(i, UP)
            self.play(FadeIn(intersection_text))

            u = Union(ellipse1, ellipse2, color=ORANGE, fill_opacity=0.5)
            union_text = Text("Union", font_size=23)
            self.play(u.animate.scale(0.3).next_to(i, DOWN, buff=union_text.height * 3))
            union_text.next_to(u, UP)
            self.play(FadeIn(union_text))

            e = Exclusion(ellipse1, ellipse2, color=YELLOW, fill_opacity=0.5)
            exclusion_text = Text("Exclusion", font_size=23)
            self.play(e.animate.scale(0.3).next_to(u, DOWN, buff=exclusion_text.height * 3.5))
            exclusion_text.next_to(e, UP)
            self.play(FadeIn(exclusion_text))

            d = Difference(ellipse1, ellipse2, color=PINK, fill_opacity=0.5)
            difference_text = Text("Difference", font_size=23)
            self.play(d.animate.scale(0.3).next_to(u, LEFT, buff=difference_text.height * 3.5))
            difference_text.next_to(d, UP)
            self.play(FadeIn(difference_text))
        """,
    },
    {
        "description": "PointMovingOnShapes",
        "code": """
        class PointMovingOnShapes(Scene):
        def construct(self):
            circle = Circle(radius=1, color=BLUE)
            dot = Dot()
            dot2 = dot.copy().shift(RIGHT)
            self.add(dot)

            line = Line([3, 0, 0], [5, 0, 0])
            self.add(line)

            self.play(GrowFromCenter(circle))
            self.play(Transform(dot, dot2))
            self.play(MoveAlongPath(dot, circle), run_time=2, rate_func=linear)
            self.play(Rotating(dot, about_point=[2, 0, 0]), run_time=1.5)
            self.wait()
        """,
    },
    {
        "description": "MovingAround",
        "code": """
        from manim import *
        class MovingAround(Scene):
            def construct(self):
                square = Square(color=BLUE, fill_opacity=1)

                self.play(square.animate.shift(LEFT))
                self.play(square.animate.set_fill(ORANGE))
                self.play(square.animate.scale(0.3))
                self.play(square.animate.rotate(0.4))
        """,
    },
    {
        "description": "MovingAngle",
        "code": """
        from manim import *

        class MovingAngle(Scene):
            def construct(self):
                rotation_center = LEFT

                theta_tracker = ValueTracker(110)
                line1 = Line(LEFT, RIGHT)
                line_moving = Line(LEFT, RIGHT)
                line_ref = line_moving.copy()
                line_moving.rotate(
                    theta_tracker.get_value() * DEGREES, about_point=rotation_center
                )
                a = Angle(line1, line_moving, radius=0.5, other_angle=False)
                tex = MathTex(r"\theta").move_to(
                    Angle(
                        line1, line_moving, radius=0.5 + 3 * SMALL_BUFF, other_angle=False
                    ).point_from_proportion(0.5)
                )

                self.add(line1, line_moving, a, tex)
                self.wait()

                line_moving.add_updater(
                    lambda x: x.become(line_ref.copy()).rotate(
                        theta_tracker.get_value() * DEGREES, about_point=rotation_center
                    )
                )

                a.add_updater(
                    lambda x: x.become(Angle(line1, line_moving, radius=0.5, other_angle=False))
                )
                tex.add_updater(
                    lambda x: x.move_to(
                        Angle(
                            line1, line_moving, radius=0.5 + 3 * SMALL_BUFF, other_angle=False
                        ).point_from_proportion(0.5)
                    )
                )

                self.play(theta_tracker.animate.set_value(40))
                self.play(theta_tracker.animate.increment_value(140))
                self.play(tex.animate.set_color(RED), run_time=0.5)
                self.play(theta_tracker.animate.set_value(350))
        """,
    },
    {
        "description": "MovingFrameBox",
        "code": """
        class MovingFrameBox(Scene):
        def construct(self):
            text=MathTex(
                "\\frac{d}{dx}f(x)g(x)=","f(x)\\frac{d}{dx}g(x)","+",
                "g(x)\\frac{d}{dx}f(x)"
            )
            self.play(Write(text))
            framebox1 = SurroundingRectangle(text[1], buff = .1)
            framebox2 = SurroundingRectangle(text[3], buff = .1)
            self.play(
                Create(framebox1),
            )
            self.wait()
            self.play(
                ReplacementTransform(framebox1,framebox2),
            )
            self.wait()
        """,
    },
    {
        "description": "SinAndCosFunctionPlot",
        "code": """
        class SinAndCosFunctionPlot(Scene):
        def construct(self):
            axes = Axes(
                x_range=[-10, 10.3, 1],
                y_range=[-1.5, 1.5, 1],
                x_length=10,
                axis_config={"color": GREEN},
                x_axis_config={
                    "numbers_to_include": np.arange(-10, 10.01, 2),
                    "numbers_with_elongated_ticks": np.arange(-10, 10.01, 2),
                },
                tips=False,
            )
            axes_labels = axes.get_axis_labels()
            sin_graph = axes.plot(lambda x: np.sin(x), color=BLUE)
            cos_graph = axes.plot(lambda x: np.cos(x), color=RED)

            sin_label = axes.get_graph_label(
                sin_graph, "\\sin(x)", x_val=-10, direction=UP / 2
            )
            cos_label = axes.get_graph_label(cos_graph, label="\\cos(x)")

            vert_line = axes.get_vertical_line(
                axes.i2gp(TAU, cos_graph), color=YELLOW, line_func=Line
            )
            line_label = axes.get_graph_label(
                cos_graph, r"x=2\pi", x_val=TAU, direction=UR, color=WHITE
            )

            plot = VGroup(axes, sin_graph, cos_graph, vert_line)
            labels = VGroup(axes_labels, sin_label, cos_label, line_label)
            self.add(plot, labels)
        """,
    },
    {
        "description": "GraphAreaPlot",
        "code": """
        class GraphAreaPlot(Scene):
        def construct(self):
            ax = Axes(
                x_range=[0, 5],
                y_range=[0, 6],
                x_axis_config={"numbers_to_include": [2, 3]},
                tips=False,
            )

            labels = ax.get_axis_labels()

            curve_1 = ax.plot(lambda x: 4 * x - x ** 2, x_range=[0, 4], color=BLUE_C)
            curve_2 = ax.plot(
                lambda x: 0.8 * x ** 2 - 3 * x + 4,
                x_range=[0, 4],
                color=GREEN_B,
            )

            line_1 = ax.get_vertical_line(ax.input_to_graph_point(2, curve_1), color=YELLOW)
            line_2 = ax.get_vertical_line(ax.i2gp(3, curve_1), color=YELLOW)

            riemann_area = ax.get_riemann_rectangles(curve_1, x_range=[0.3, 0.6], dx=0.03, color=BLUE, fill_opacity=0.5)
            area = ax.get_area(curve_2, [2, 3], bounded_graph=curve_1, color=GREY, opacity=0.5)

            self.add(ax, labels, curve_1, curve_2, line_1, line_2, riemann_area, area)
        
        """,
    },
    {
        "description": "PolygonOnAxes",
        "code": """
            from manim import *

            class PolygonOnAxes(Scene):
                def get_rectangle_corners(self, bottom_left, top_right):
                    return [
                        (top_right[0], top_right[1]),
                        (bottom_left[0], top_right[1]),
                        (bottom_left[0], bottom_left[1]),
                        (top_right[0], bottom_left[1]),
                    ]

                def construct(self):
                    ax = Axes(
                        x_range=[0, 10],
                        y_range=[0, 10],
                        x_length=6,
                        y_length=6,
                        axis_config={"include_tip": False},
                    )

                    t = ValueTracker(5)
                    k = 25

                    graph = ax.plot(
                        lambda x: k / x,
                        color=YELLOW_D,
                        x_range=[k / 10, 10.0, 0.01],
                        use_smoothing=False,
                    )

                    def get_rectangle():
                        polygon = Polygon(
                            *[
                                ax.c2p(*i)
                                for i in self.get_rectangle_corners(
                                    (0, 0), (t.get_value(), k / t.get_value())
                                )
                            ]
                        )
                        polygon.stroke_width = 1
                        polygon.set_fill(BLUE, opacity=0.5)
                        polygon.set_stroke(YELLOW_B)
                        return polygon

                    polygon = always_redraw(get_rectangle)

                    dot = Dot()
                    dot.add_updater(lambda x: x.move_to(ax.c2p(t.get_value(), k / t.get_value())))
                    dot.set_z_index(10)

                    self.add(ax, graph, dot)
                    self.play(Create(polygon))
                    self.play(t.animate.set_value(10))
                    self.play(t.animate.set_value(k / 10))
                    self.play(t.animate.set_value(5))
        """,
    },
    {
        "description": "OpeningManim",
        "code": """
        class OpeningManim(Scene):
    def construct(self):
        title = Tex(r"This is some \LaTeX")
        basel = MathTex(r"\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}")
        VGroup(title, basel).arrange(DOWN)
        self.play(
            Write(title),
            FadeIn(basel, shift=DOWN),
        )
        self.wait()

        transform_title = Tex("That was a transform")
        transform_title.to_corner(UP + LEFT)
        self.play(
            Transform(title, transform_title),
            LaggedStart(*[FadeOut(obj, shift=DOWN) for obj in basel]),
        )
        self.wait()

        grid = NumberPlane()
        grid_title = Tex("This is a grid", font_size=72)
        grid_title.move_to(transform_title)

        self.add(grid, grid_title)  # Make sure title is on top of grid
        self.play(
            FadeOut(title),
            FadeIn(grid_title, shift=UP),
            Create(grid, run_time=3, lag_ratio=0.1),
        )
        self.wait()

        grid_transform_title = Tex(
            r"That was a non-linear function \\ applied to the grid"
        )
        grid_transform_title.move_to(grid_title, UL)
        grid.prepare_for_nonlinear_transform()
        self.play(
            grid.animate.apply_function(
                lambda p: p
                          + np.array(
                    [
                        np.sin(p[1]),
                        np.sin(p[0]),
                        0,
                    ]
                )
            ),
            run_time=3,
        )
        self.wait()
        self.play(Transform(grid_title, grid_transform_title))
        self.wait()
        """,
    },
    {
        "description": "SineCurveUnitCircle",
        "code": """
        from manim import *

class SineCurveUnitCircle(Scene):
    # contributed by heejin_park, https://infograph.tistory.com/230
    def construct(self):
        self.show_axis()
        self.show_circle()
        self.move_dot_and_draw_curve()
        self.wait()

    def show_axis(self):
        x_start = np.array([-6,0,0])
        x_end = np.array([6,0,0])

        y_start = np.array([-4,-2,0])
        y_end = np.array([-4,2,0])

        x_axis = Line(x_start, x_end)
        y_axis = Line(y_start, y_end)

        self.add(x_axis, y_axis)
        self.add_x_labels()

        self.origin_point = np.array([-4,0,0])
        self.curve_start = np.array([-3,0,0])

    def add_x_labels(self):
        x_labels = [
            MathTex(r"\pi"), MathTex(r"2 \pi"),
            MathTex(r"3 \pi"), MathTex(r"4 \pi"),
        ]

        for i in range(len(x_labels)):
            x_labels[i].next_to(np.array([-1 + 2*i, 0, 0]), DOWN)
            self.add(x_labels[i])

    def show_circle(self):
        circle = Circle(radius=1)
        circle.move_to(self.origin_point)
        self.add(circle)
        self.circle = circle

    def move_dot_and_draw_curve(self):
        orbit = self.circle
        origin_point = self.origin_point

        dot = Dot(radius=0.08, color=YELLOW)
        dot.move_to(orbit.point_from_proportion(0))
        self.t_offset = 0
        rate = 0.25

        def go_around_circle(mob, dt):
            self.t_offset += (dt * rate)
            # print(self.t_offset)
            mob.move_to(orbit.point_from_proportion(self.t_offset % 1))

        def get_line_to_circle():
            return Line(origin_point, dot.get_center(), color=BLUE)

        def get_line_to_curve():
            x = self.curve_start[0] + self.t_offset * 4
            y = dot.get_center()[1]
            return Line(dot.get_center(), np.array([x,y,0]), color=YELLOW_A, stroke_width=2 )


        self.curve = VGroup()
        self.curve.add(Line(self.curve_start,self.curve_start))
        def get_curve():
            last_line = self.curve[-1]
            x = self.curve_start[0] + self.t_offset * 4
            y = dot.get_center()[1]
            new_line = Line(last_line.get_end(),np.array([x,y,0]), color=YELLOW_D)
            self.curve.add(new_line)

            return self.curve

        dot.add_updater(go_around_circle)

        origin_to_circle_line = always_redraw(get_line_to_circle)
        dot_to_curve_line = always_redraw(get_line_to_curve)
        sine_curve_line = always_redraw(get_curve)

        self.add(dot)
        self.add(orbit, origin_to_circle_line, dot_to_curve_line, sine_curve_line)
        self.wait(8.5)

        dot.remove_updater(go_around_circle)
        
        """,
    },
]

rules = [
    """
    1. in construct method, first set the default font to source code pro
    def construct(self):
        Text.set_default(font="Source Code Pro")
    """,
    """
    2. do NOT cluster everything in one scene. Its not gonna display properly. compose scenes together like below. each scene corresponds to one small animation
    # Helper function to fade out all mobjects in a scene
    def fade_out(scene: Scene):
        if not scene.mobjects:
            return
        animations = []
        for mobject in scene.mobjects:
            animations.append(FadeOut(mobject))
        scene.play(*animations)
 
    class CombinedScene(Scene):
        def construct(self):
            scenes: list[Scene] = [
                ProblemIntroduction,
                IndicatorVisualization,
                CalculationScene,
            ]
            for scene in scenes:
                scene.construct(self)
                fade_out(self)
                self.wait(0.5)
    """,
    """
    3. use visualizations to help illustrate the concepts to make them understandable by people.
    """,
]


def clean_str(s: str) -> str:
    # clean up whitespaces, newlines to reduce token usage
    return " ".join(s.split())


def build_prompt(question: str) -> str:
    assert len(examples) > 0, "No examples provided"
    assert len(rules) > 0, "No rules provided"
    assert question is not None, "Question is required"
    role_def = """You are a specialized agent in Manim, a python library for creating animation videos. You will be generating manim code that produces illustrative, concise, and effective animations that answers the questions or solving the interview questions. You will be specifically given questions related in quant finance, most of which are probability/statistics related. You will be given some samples snippets of using Manim, and you will generate code that illustrates the core ideas of solving the problems.
    """

    example_str = "".join(
        [f"## {example['description']}\n{example['code']}\n" for example in examples]
    )
    rules_str = "\n".join(rules)

    prompt = f"""{role_def}
    Here are some code examples of using manim:
    # BEGIN MANIM USAGE EXAMPLES
    {example_str}
    # END MANIM USAGE EXAMPLES
    
    Here are some additional rules. This is important to my career.
    {rules_str}
    
    you will also be given technical questions, possibly with solutions. your goal is to produce effective manim code that illustrates the core idea and explains the solution to people. we will make video using the produced code.
    
    question: {question}
    
    """
    return clean_str(prompt)
