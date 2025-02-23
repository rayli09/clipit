from manim import *


# Helper function to fade out all mobjects in a scene
def fade_out(scene: Scene):
    if not scene.mobjects:
        return
    animations = []
    for mobject in scene.mobjects:
        animations.append(FadeOut(mobject))
    scene.play(*animations)


# Scene 1: Introduce the Problem
class ProblemIntroduction(Scene):
    def construct(self):
        Text.set_default(font="Helvetica Neue")

        # Title and problem statement
        title = Text("Elevator Floor Stops Problem", font_size=36)
        statement = Text(
            "12 people enter an elevator at the basement.\n"
            "There are 10 floors above.\n"
            "Each person chooses a floor uniformly at random.\n"
            "How many floors does the elevator stop at?",
            font_size=28,
            t2c={"12": YELLOW, "10": YELLOW},
        )
        statement.next_to(title, DOWN, buff=0.5)
        self.play(Write(title))
        self.wait(1)
        self.play(Write(statement))
        self.wait(2)

        # Visual: Draw a building with 10 floors (simplified as 10 boxes)
        building = VGroup()
        for i in range(10):
            floor = Square(side_length=0.6, color=BLUE)
            floor.move_to(np.array([-4, i * 0.7 - 3, 0]))
            floor_text = Text(f"Floor {i + 1}", font_size=20)
            floor_text.move_to(floor.get_center())
            building.add(VGroup(floor, floor_text))
        self.play(LaggedStart(*[FadeIn(mob) for mob in building], lag_ratio=0.1))
        self.wait(2)
        self.add(building)
        self.wait(2)


# Scene 2: Visualizing Indicator Variables
class IndicatorVisualization(Scene):
    def construct(self):
        Text.set_default(font="Helvetica Neue")

        # Title for this scene
        title = Text("Indicator Variables per Floor", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # Explanation text
        explanation = Text(
            "For each floor i, define X_i = 1 if at least one person gets off at floor i,\n"
            "and 0 otherwise.",
            font_size=28,
        )
        explanation.next_to(title, DOWN, buff=0.5)
        self.play(Write(explanation))
        self.wait(2)

        # Visual: Represent 10 floors with indicators
        indicators = VGroup()
        for i in range(10):
            # Create a box for the floor indicator
            box = Square(side_length=0.8, color=WHITE)
            box.shift(np.array([-4 + (i % 5) * 1.8, -1 - (i // 5) * 2, 0]))
            label = Text(f"X_{i + 1}", font_size=24)
            label.move_to(box.get_center())
            indicators.add(VGroup(box, label))
        self.play(LaggedStart(*[FadeIn(ind) for ind in indicators], lag_ratio=0.2))
        self.wait(2)

        # Show probability expression on the side
        prob_text = MathTex(
            r"P(X_i=1)=1-P(\text{no one gets off at floor } i)=1-\left(\frac{9}{10}\right)^{12}"
        ).scale(0.8)
        prob_text.to_edge(RIGHT)
        self.play(FadeIn(prob_text))
        self.wait(3)


# Scene 3: Final Calculation and Conclusion
class CalculationScene(Scene):
    def construct(self):
        Text.set_default(font="Helvetica Neue")

        # Title for final scene
        title = Text("Expected Number of Stops", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # Step-by-step calculation explanation
        step1 = MathTex(
            r"E[S]=E\left[\sum_{i=1}^{10}X_i\right]=\sum_{i=1}^{10}E[X_i]"
        ).scale(0.9)
        step1.next_to(title, DOWN, buff=0.5)
        self.play(Write(step1))
        self.wait(2)

        step2 = MathTex(r"E[X_i]=1-\left(\frac{9}{10}\right)^{12}").scale(0.9)
        step2.next_to(step1, DOWN, buff=0.7)
        self.play(Write(step2))
        self.wait(2)

        step3 = MathTex(r"E[S]=10\left(1-\left(\frac{9}{10}\right)^{12}\right)").scale(
            0.9
        )
        step3.next_to(step2, DOWN, buff=0.7)
        self.play(Write(step3))
        self.wait(2)

        # Final numeric evaluation
        final_value = MathTex(r"E[S]\approx 7.17").scale(1.0)
        final_value.next_to(step3, DOWN, buff=0.7)
        self.play(Write(final_value))
        self.wait(3)

        # Verbal conclusion
        conclusion = Text(
            "Thus, the elevator is expected to stop at around 7 floors.", font_size=28
        )
        conclusion.next_to(final_value, DOWN, buff=0.7)
        self.play(FadeIn(conclusion))
        self.wait(3)


# Combined Scene: Plays all scenes in sequence with fade outs
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
