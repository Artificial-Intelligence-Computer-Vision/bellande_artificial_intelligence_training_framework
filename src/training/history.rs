// Copyright (C) 2024 Bellande Artificial Intelligence Computer Vision Research Innovation Center, Ronaldson Bellande

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

pub struct TrainingHistory {
    metrics: HashMap<String, Vec<f32>>,
    epochs: Vec<usize>,
}

impl TrainingHistory {
    pub fn plot(&self, metric_names: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        // Setup the output path
        let output_path = PathBuf::from("training_history.png");

        // Calculate the dimensions based on number of metrics
        let num_metrics = metric_names.len();
        let num_rows = (num_metrics as f32).sqrt().ceil() as u32;
        let num_cols = ((num_metrics as f32) / num_rows as f32).ceil() as u32;

        // Setup the drawing area with multiple plots
        let root =
            BitMapBackend::new(&output_path, (800 * num_cols, 400 * num_rows)).into_drawing_area();

        root.fill(&WHITE)?;

        let areas = root.split_evenly((num_rows as usize, num_cols as usize));

        // Plot each metric
        for (area, &metric_name) in areas.iter().zip(metric_names.iter()) {
            if let Some(values) = self.get_metric_history(metric_name) {
                self.plot_metric(area, metric_name, values)?;
            }
        }

        root.present()?;

        Ok(())
    }

    fn plot_metric(
        &self,
        area: &DrawingArea<BitMapBackend, Shift>,
        metric_name: &str,
        values: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Calculate value range with some padding
        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let range_padding = (max_val - min_val) * 0.1;
        let y_range = Range {
            start: (min_val - range_padding) as f64,
            end: (max_val + range_padding) as f64,
        };

        // Configure chart
        let mut chart = ChartBuilder::on(area)
            .caption(metric_name, ("sans-serif", 20).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0..values.len(), y_range)?;

        // Configure grid and labels
        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .draw()?;

        // Plot the metric values
        chart
            .draw_series(LineSeries::new(
                values.iter().enumerate().map(|(x, &y)| (x, y as f64)),
                &BLUE,
            ))?
            .label(metric_name)
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        // Add validation metric if it exists
        let val_metric_name = format!("val_{}", metric_name);
        if let Some(val_values) = self.get_metric_history(&val_metric_name) {
            chart
                .draw_series(LineSeries::new(
                    val_values.iter().enumerate().map(|(x, &y)| (x, y as f64)),
                    &RED,
                ))?
                .label(&val_metric_name)
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        }

        // Draw legend
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        Ok(())
    }

    pub fn plot_to_file(
        &self,
        metric_names: &[&str],
        output_path: &str,
        size: (u32, u32),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(output_path, size).into_drawing_area();
        root.fill(&WHITE)?;

        let num_metrics = metric_names.len();
        let areas = root.split_evenly((
            (num_metrics as f32).sqrt().ceil() as usize,
            ((num_metrics as f32).sqrt()).ceil() as usize,
        ));

        for (area, &metric_name) in areas.iter().zip(metric_names.iter()) {
            if let Some(values) = self.get_metric_history(metric_name) {
                self.plot_metric(area, metric_name, values)?;
            }
        }

        root.present()?;

        Ok(())
    }

    pub fn plot_combined(
        &self,
        metric_names: &[&str],
        output_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();

        root.fill(&WHITE)?;

        // Find global value range
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for &metric_name in metric_names {
            if let Some(values) = self.get_metric_history(metric_name) {
                min_val = min_val.min(values.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
                max_val = max_val.max(values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
            }
        }

        let range_padding = (max_val - min_val) * 0.1;
        let y_range = Range {
            start: (min_val - range_padding) as f64,
            end: (max_val + range_padding) as f64,
        };

        // Create chart
        let mut chart = ChartBuilder::on(&root)
            .caption("Training Metrics", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0..self.epochs.len(), y_range)?;

        chart
            .configure_mesh()
            .x_desc("Epoch")
            .y_desc("Value")
            .draw()?;

        // Plot each metric with a different color
        let colors = [
            &BLUE,
            &RED,
            &GREEN,
            &CYAN,
            &MAGENTA,
            &YELLOW,
            &RGBColor(150, 75, 0),
            &RGBColor(75, 0, 150),
            &RGBColor(150, 150, 0),
            &RGBColor(0, 150, 150),
        ];

        for (i, &metric_name) in metric_names.iter().enumerate() {
            if let Some(values) = self.get_metric_history(metric_name) {
                let color = colors[i % colors.len()];

                chart
                    .draw_series(LineSeries::new(
                        values.iter().enumerate().map(|(x, &y)| (x, y as f64)),
                        color,
                    ))?
                    .label(metric_name)
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
            }
        }

        // Draw legend
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;

        root.present()?;

        Ok(())
    }
}
