mod imgproc;

fn main() {
    
    let image_path = "/Users/kumada/Pictures/dreamtheater.jpg";
    let scale = 1.1;
    
    let img = image::open(image_path).unwrap().to_rgb8();
    let dst = imgproc::resize_with_nearest_neighbor(&img, scale); 
    dst.save("./outputs/nearest.jpg").unwrap();

    let img = image::open(image_path).unwrap().to_rgb32f();
    let dst = imgproc::resize_with_bilinear(&img, scale); 
    dst.save("./outputs/bilinear.jpg").unwrap();
}
