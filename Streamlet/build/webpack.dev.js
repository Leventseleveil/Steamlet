const { merge } = require('webpack-merge');
const common = require('./webpack.base.js').config;
const webpack = require("webpack");
const path = require('path');

const config = merge(common, {
    mode: 'development',
    entry: {
        'dash.all': './index.js',
        'dash.mss': './src/mss/index.js',
        'dash.offline': './src/offline/index.js'
    },
    output: {
        filename: '[name].debug.js',
    },
    devServer: {
        contentBase: path.join(__dirname, '../'),
        open: true,
        openPage: 'samples/streamlet/testplayer.html',
        hot: true,
        compress: true,
        host: "10.181.147.245",
        port: 3000,
        watchOptions: {
            aggregateTimeout: 300,
            poll: 1000
        }
    },
    plugins: [
        new webpack.HotModuleReplacementPlugin(),
    ]
});

module.exports = config;
