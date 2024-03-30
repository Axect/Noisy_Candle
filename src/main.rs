use peroxide::fuga::*;
use candle_core::{DType, Device, Result, Tensor, Module};
use candle_nn::{Linear, VarBuilder, linear, VarMap, Optimizer, loss};
use candle_optimisers::adam::{Adam, ParamsAdam};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

const SEED: u64 = 42;

fn main() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;

    let ds = Dataset::<RobustScaler>::generate(1800, 200, 1000, &dev)?;

    let model = train(ds.clone(), &dev)?;
    test(ds.clone(), &model, &dev)?;

    println!("Done");

    let (test_x, test_y) = ds.test_set(&dev)?;
    let y_hat = model.forward(&test_x)?;

    let test_x = test_x.detach().squeeze(1)?.to_vec1()?;
    let test_y = test_y.detach().squeeze(1)?.to_vec1()?;
    let y_hat = y_hat.detach().squeeze(1)?.to_vec1()?;

    let scaler_x = ds.scaler_x;
    let scaler_y = ds.scaler_y;

    let test_x = scaler_x.unscale_vec(&test_x);
    let test_y = scaler_y.unscale_vec(&test_y);
    let y_hat = scaler_y.unscale_vec(&y_hat);

    // sort by x
    let mut test_ics_x = test_x.into_iter().enumerate().collect::<Vec<_>>();
    test_ics_x.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let (test_ics, test_x): (Vec<_>, Vec<_>) = test_ics_x.into_iter().unzip();
    let test_x = test_x.into_iter().map(|x| x as f64).collect::<Vec<_>>();
    let (test_y, y_hat): (Vec<_>, Vec<_>) = test_ics.into_iter()
        .map(|i| (test_y[i] as f64, y_hat[i] as f64))
        .unzip();

    let mut plt = Plot2D::new();
    plt
        .set_domain(test_x)
        .insert_image(test_y)
        .insert_image(y_hat)
        .set_marker(vec![(0, Markers::Point)])
        .set_plot_type(vec![(0, PlotType::Scatter)])
        .set_style(PlotStyle::Nature)
        .set_color(vec![(1, "red")])
        .set_alpha(vec![(0, 0.5), (1, 1.0)])
        .set_legend(vec!["Data", "Model"])
        .set_xlabel(r"$x$")
        .set_ylabel(r"$y$")
        .tight_layout()
        .set_dpi(600)
        .set_path("test_plot.png")
        .savefig().unwrap();


    Ok(())
}

// ┌─────────────────────────────────────────────────────────┐
//  Dataset
// └─────────────────────────────────────────────────────────┘
/// Dataset for Noisy Regression
#[derive(Clone)]
pub struct Dataset<S: Scaler> {
    pub train_x: Tensor,
    pub train_y: Tensor,
    pub val_x: Tensor,
    pub val_y: Tensor,
    pub test_x: Tensor,
    pub test_y: Tensor,
    pub scaler_x: S,
    pub scaler_y: S
}

impl<S: Scaler> Dataset<S> {
    pub fn generate(n_train: usize, n_val: usize, n_test: usize, device: &Device) -> Result<Self> {
        let noise = Normal(0., 0.5);
        let p_true = vec![20f64, 10f64, 1f64, 50f64];

        let domain = linspace(0, 100, n_train + n_val + n_test);
        let y = f(&domain, &p_true).add_v(&noise.sample(domain.len()));

        // Randomly split data into train/val/test
        let mut ics = (0 .. n_train + n_val + n_test).collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(SEED);
        ics.shuffle(&mut rng);

        let train_ics = &ics[0 .. n_train];
        let val_ics = &ics[n_train .. n_train + n_val];
        let test_ics = &ics[n_train + n_val .. n_train + n_val + n_test];

        let (train_x, train_y): (Vec<f32>, Vec<f32>) = train_ics.iter()
            .map(|&i| (domain[i] as f32, y[i] as f32))
            .unzip();
        let (val_x, val_y): (Vec<f32>, Vec<f32>) = val_ics.iter()
            .map(|&i| (domain[i] as f32, y[i] as f32))
            .unzip();
        let (test_x, test_y): (Vec<f32>, Vec<f32>) = test_ics.iter()
            .map(|&i| (domain[i] as f32, y[i] as f32))
            .unzip();

        let scaler_x = S::new(&train_x);
        let scaler_y = S::new(&train_y);

        let train_x = scaler_x.scale_vec(&train_x);
        let train_y = scaler_y.scale_vec(&train_y);
        let val_x = scaler_x.scale_vec(&val_x);
        let val_y = scaler_y.scale_vec(&val_y);
        let test_x = scaler_x.scale_vec(&test_x);
        let test_y = scaler_y.scale_vec(&test_y);

        Ok(Self {
            train_x: Tensor::from_vec(train_x, (n_train, 1), device)?,
            train_y: Tensor::from_vec(train_y, (n_train, 1), device)?,
            val_x: Tensor::from_vec(val_x, (n_val, 1), device)?,
            val_y: Tensor::from_vec(val_y, (n_val, 1), device)?,
            test_x: Tensor::from_vec(test_x, (n_test, 1), device)?,
            test_y: Tensor::from_vec(test_y, (n_test, 1), device)?,
            scaler_x,
            scaler_y,
        })
    }

    pub fn train_set(&self, dev: &Device) -> Result<(Tensor, Tensor)> {
        Ok((self.train_x.to_device(dev)?, self.train_y.to_device(dev)?))
    }

    pub fn val_set(&self, dev: &Device) -> Result<(Tensor, Tensor)> {
        Ok((self.val_x.to_device(dev)?, self.val_y.to_device(dev)?))
    }

    pub fn test_set(&self, dev: &Device) -> Result<(Tensor, Tensor)> {
        Ok((self.test_x.to_device(dev)?, self.test_y.to_device(dev)?))
    }

    pub fn scaler_x(&self) -> &S {
        &self.scaler_x
    }

    pub fn scaler_y(&self) -> &S {
        &self.scaler_y
    }

    pub fn all_set(&self, dev: &Device) -> Result<(Tensor, Tensor)> {
        let (train_x, train_y) = self.train_set(dev)?;
        let (val_x, val_y) = self.val_set(dev)?;
        let (test_x, test_y) = self.test_set(dev)?;

        let all_x = Tensor::stack(&[&train_x, &val_x, &test_x], 0)?;
        let all_y = Tensor::stack(&[&train_y, &val_y, &test_y], 0)?;

        Ok((all_x, all_y))
    }
}

fn f(domain: &[f64], p: &[f64]) -> Vec<f64> {
    domain.par_iter()
        .map(|x| p[0] * (-x / p[1]).exp() + p[2] * x * (-x / p[3]).exp())
        .collect()
}

// ┌─────────────────────────────────────────────────────────┐
//  Scaler
// └─────────────────────────────────────────────────────────┘
pub trait Scaler: Sized + Copy + Clone + Sync {
    fn new(x: &[f32]) -> Self;
    fn scale(&self, x: f32) -> f32;
    fn unscale(&self, x: f32) -> f32;

    fn scale_vec(&self, x: &[f32]) -> Vec<f32> {
        x.par_iter().map(|&x| self.scale(x)).collect()
    }

    fn unscale_vec(&self, x: &[f32]) -> Vec<f32> {
        x.par_iter().map(|&x| self.unscale(x)).collect()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct MinMaxScaler {
    pub min: f32,
    pub max: f32
}

impl Scaler for MinMaxScaler {
    fn new(x: &[f32]) -> Self {
        let (min, max) = x.iter()
            .fold((x[0], x[0]), |(min, max), &x| (min.min(x), max.max(x)));
        Self { min, max }
    }

    fn scale(&self, x: f32) -> f32 {
        (x - self.min) / (self.max - self.min)
    }

    fn unscale(&self, x: f32) -> f32 {
        x * (self.max - self.min) + self.min
    }
}

#[derive(Debug, Copy, Clone)]
pub struct StandardScaler {
    pub mean: f32,
    pub std: f32
}

impl Scaler for StandardScaler {
    fn new(x: &[f32]) -> Self {
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let std = (x.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / x.len() as f32).sqrt();
        Self { mean, std }
    }

    fn scale(&self, x: f32) -> f32 {
        (x - self.mean) / self.std
    }

    fn unscale(&self, x: f32) -> f32 {
        x * self.std + self.mean
    }
}

#[derive(Debug, Copy, Clone)]
pub struct RobustScaler {
    pub median: f32,
    pub mad: f32
}

impl Scaler for RobustScaler {
    fn new(x: &[f32]) -> Self {
        let median = x.iter().sum::<f32>() / x.len() as f32;
        let mad = x.iter().map(|&x| (x - median).abs()).sum::<f32>() / x.len() as f32;
        Self { median, mad }
    }

    fn scale(&self, x: f32) -> f32 {
        (x - self.median) / self.mad
    }

    fn unscale(&self, x: f32) -> f32 {
        x * self.mad + self.median
    }
}

#[derive(Debug, Copy, Clone)]
pub struct IdentityScaler;

impl Scaler for IdentityScaler {
    fn new(_: &[f32]) -> Self { Self }
    fn scale(&self, x: f32) -> f32 { x }
    fn unscale(&self, x: f32) -> f32 { x }
}

// ┌─────────────────────────────────────────────────────────┐
//  Neural Network
// └─────────────────────────────────────────────────────────┘
pub struct MLP {
    lns: Vec<Linear>
}

#[derive(Debug, Copy, Clone)]
pub struct HyperParams {
    pub hidden_size: usize,
    pub hidden_depth: usize,
    pub learning_rate: f64,
    pub epoch: usize
}

impl MLP {
    pub fn new(vs: VarBuilder, hparam: HyperParams) -> Result<Self> {
        let hidden_size = hparam.hidden_size;
        let hidden_depth = hparam.hidden_depth;

        let mut lns = vec![linear(1, hidden_size, vs.pp("ln0"))?];
        for i in 1 .. hidden_depth {
            lns.push(linear(hidden_size, hidden_size, vs.pp(&format!("ln{}", i)))?);
        }

        lns.push(linear(hidden_size, 1, vs.pp("out"))?);

        Ok(Self { lns })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for ln in self.lns.iter().take(self.lns.len() - 1) {
            xs = ln.forward(&xs)?;
            xs = xs.gelu()?;
        }
        self.lns.last().unwrap().forward(&xs)
    }
}

// ┌─────────────────────────────────────────────────────────┐
//  Train
// └─────────────────────────────────────────────────────────┘
/// Train the model
pub fn train<S: Scaler>(ds: Dataset<S>, dev: &Device) -> Result<MLP> {
    let (train_x, train_y) = ds.train_set(dev)?;
    let (val_x, val_y) = ds.val_set(dev)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);

    let hparam = HyperParams {
        hidden_size: 16,
        hidden_depth: 4,
        learning_rate: 1e-3,
        epoch: 1000
    };
    let model = MLP::new(vs, hparam)?;

    let adam_param = ParamsAdam {
        lr: hparam.learning_rate,
        beta_1: 0.9,
        beta_2: 0.999,
        eps: 1e-8,
        weight_decay: None,
        amsgrad: false
    };
    let mut adam = Adam::new(varmap.all_vars(), adam_param)?; 

    let mut train_loss = 0f32;
    let mut val_loss = 0f32;

    let pb = ProgressBar::new(hparam.epoch as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
        .unwrap()
        .progress_chars("##-"));

    for epoch in 0 .. hparam.epoch {
        pb.set_position(epoch as u64);
        let msg = format!("epoch: {}, train_loss: {:.4e}, val_loss: {:.4e}", epoch, train_loss, val_loss);
        pb.set_message(msg);

        let y_hat = model.forward(&train_x)?;
        let loss = loss::mse(&y_hat, &train_y)?;
        adam.backward_step(&loss)?;
        train_loss = loss.to_scalar()?;

        let y_hat = model.forward(&val_x)?;
        let loss = loss::mse(&y_hat, &val_y)?;
        val_loss = loss.to_scalar()?;
    }

    println!("train_loss: {:.4e}, val_loss: {:.4e}", train_loss, val_loss);

    Ok(model)
}

// ┌─────────────────────────────────────────────────────────┐
//  Test
// └─────────────────────────────────────────────────────────┘
pub fn test<S: Scaler>(ds: Dataset<S>, model: &MLP, dev: &Device) -> Result<()> {
    let (test_x, test_y) = ds.test_set(dev)?;

    let y_hat = model.forward(&test_x)?;
    let loss = loss::mse(&y_hat, &test_y)?;
    let loss_f32: f32 = loss.to_scalar()?;
    println!("test_loss: {:.4e}", loss_f32);
    Ok(())
}
